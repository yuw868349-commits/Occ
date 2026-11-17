//! Pseudo-C rendering.
//!
//! The output is meant to be read, not compiled: names are kept, casts are
//! explicit, and anything the lifter could not model appears as a `//` comment
//! at the exact place it happened. Nothing is invented to make a function look
//! complete.

use std::collections::HashMap;
use std::fmt::Write;

use crate::ir::{Decompiled, Expr, Node, type_name};

/// Render a decompiled function as pseudo-C.
pub fn render(d: &Decompiled) -> String {
    let mut out = String::new();
    let params = if d.params.is_empty() {
        "void".to_string()
    } else {
        d.params
            .iter()
            .enumerate()
            .map(|(i, p)| match d.param_types.get(i).and_then(|t| t.as_deref()) {
                // `struct point *p`, not `struct point * p`: the star belongs
                // to the declarator in C, and the debug info already spaced it.
                Some(t) if t.ends_with('*') => format!("{t}{p}"),
                Some(t) => format!("{t} {p}"),
                None => format!("u64 {p}"),
            })
            .collect::<Vec<_>>()
            .join(", ")
    };
    // The return type is only known when the body says where the function
    // left its value, or when debug info declares it. Without either, the
    // machine word is the honest answer.
    let ret = d.ret_type.as_deref().unwrap_or("u64");
    let _ = writeln!(out, "{ret} {}({params}) {{", d.name);
    let mut r = Renderer {
        out,
        indent: 1,
        aliases: Aliases::new(d),
        last_call: None,
    };
    if let Some(src) = &d.source {
        r.line(&format!(
            "// source: {}:{}-{}",
            src.file, src.first_line, src.last_line
        ));
    }
    r.nodes(&d.body);
    for note in &d.notes {
        r.line(&format!("// note: {note}"));
    }
    r.out.push_str("}\n");
    r.out
}

struct Renderer {
    out: String,
    indent: usize,
    aliases: Aliases,
    /// Where the previous statement started, together with the call it printed
    /// when that statement was `dest = callee(args);`. Returning `dest` right
    /// after it is the same thing the source said in one line — a `return` of a
    /// call's result, which is how tail calls come out — so the two are printed
    /// as one rather than as a temporary the reader has to follow.
    last_call: Option<(usize, String, Vec<Expr>, String)>,
}

impl Renderer {
    fn line(&mut self, text: &str) {
        for _ in 0..self.indent {
            self.out.push_str("    ");
        }
        self.out.push_str(text);
        self.out.push('\n');
    }

    fn open(&mut self, text: &str) {
        self.line(&format!("{text} {{"));
        self.indent += 1;
    }

    fn close(&mut self, suffix: &str) {
        self.indent = self.indent.saturating_sub(1);
        self.line(&format!("}}{suffix}"));
    }

    fn nodes(&mut self, nodes: &[Node]) {
        for n in nodes {
            self.node(n);
        }
    }

    fn node(&mut self, node: &Node) {
        // Anything that is not a call ends the run in which a following
        // `return` could still be folded into it.
        let previous_call = self.last_call.take();
        match node {
            Node::Assign { dest, value } => {
                let line = format!("{} = {};", name(dest), expr(value));
                self.line(&line);
            }
            Node::Store { addr, value, size } => {
                let line = format!(
                    "*({}*){} = {};",
                    type_name(*size, false),
                    expr(addr),
                    expr(value)
                );
                self.line(&line);
            }
            Node::Call { name: callee, args, ret } => {
                let call = format!(
                    "{}({})",
                    callee,
                    args.iter().map(expr).collect::<Vec<_>>().join(", ")
                );
                match ret {
                    Some(dest) => {
                        let at = self.out.len();
                        let line = format!("{} = {call};", name(dest));
                        self.line(&line);
                        self.last_call = Some((at, callee.clone(), args.clone(), dest.clone()));
                    }
                    None => self.line(&format!("{call};")),
                }
            }
            Node::If {
                cond,
                then_body,
                else_body,
            } => {
                self.open(&format!("if ({})", expr(cond)));
                self.nodes(then_body);
                self.indent = self.indent.saturating_sub(1);
                if else_body.is_empty() {
                    self.line("}");
                } else {
                    self.line("} else {");
                    self.indent += 1;
                    self.nodes(else_body);
                    self.indent = self.indent.saturating_sub(1);
                    self.line("}");
                }
            }
            Node::While { cond, body } => {
                self.open(&format!("while ({})", expr(cond)));
                self.nodes(body);
                self.close("");
            }
            Node::DoWhile { body, cond } => {
                self.open("do");
                self.nodes(body);
                self.close(&format!(" while ({});", expr(cond)));
            }
            Node::Switch {
                expr: e,
                cases,
                default,
            } => {
                self.open(&format!("switch ({})", expr(e)));
                for (value, body) in cases {
                    self.indent = self.indent.saturating_sub(1);
                    self.line(&format!("case {value:#x}:"));
                    self.indent += 1;
                    self.nodes(body);
                }
                if !default.is_empty() {
                    self.indent = self.indent.saturating_sub(1);
                    self.line("default:");
                    self.indent += 1;
                    self.nodes(default);
                }
                self.close("");
            }
            Node::Return(Some(value)) => {
                if let (Expr::Var(v), Some((at, callee, args, dest))) = (value, previous_call)
                    && *v == *dest
                {
                    self.out.truncate(at);
                    let call = format!(
                        "{callee}({})",
                        args.iter().map(expr).collect::<Vec<_>>().join(", ")
                    );
                    self.line(&format!("return {call};"));
                    return;
                }
                let line = format!("return {};", expr(value));
                self.line(&line);
            }
            Node::Return(None) => self.line("return;"),
            Node::Goto(l) => self.line(&format!("goto {l};")),
            // A jump through a value has no label to name, and printing the
            // value is the only honest thing left: the reader can see which
            // register or cell the dispatch went through.
            Node::ComputedGoto(e) => self.line(&format!("goto *({});", expr(e))),
            Node::Label(l) => {
                self.indent = self.indent.saturating_sub(1);
                self.line(&format!("{l}:"));
                self.indent += 1;
            }
            Node::Break => self.line("break;"),
            Node::Continue => self.line("continue;"),
            Node::Nop => {}
            Node::Comment(c) => self.line(&format!("// {c}")),
        }
    }

    fn fmt_expr(&self, e: &Expr, out: &mut String, min: u8) {
    let (p, needs) = prec(e);
    let paren = needs && p < min;
    if paren {
        out.push('(');
    }
    match e {
        Expr::Const(v) => {
            if *v < 10 {
                let _ = write!(out, "{v}");
            } else {
                let _ = write!(out, "{v:#x}");
            }
        }
        Expr::Var(v) => out.push_str(&self.aliases.var_name(v)),
        Expr::Member { base, name, .. } => {
            // The debug info named the aggregate and the member, so the load
            // reads as the access the source wrote. A base that is not a bare
            // variable keeps its parentheses: `(a + b)->x` is still clear,
            // `a + b->x` would not be.
            let compound = !matches!(base.as_ref(), Expr::Var(_));
            if compound {
                out.push('(');
            }
            self.fmt_expr(base, out, 0);
            if compound {
                out.push(')');
            }
            let _ = write!(out, "->{name}");
        }
        Expr::Load { addr, size, signed } => {
            let _ = write!(out, "*({}*)", type_name(*size, *signed));
            self.fmt_expr(addr, out, 7);
        }
        Expr::Bin { op, lhs, rhs, bits } => {
            if let Some(name) = op.float_name(*bits) {
                // The same spelling as the expression's own: a floating-point
                // operation is a call, because its operands are bit patterns.
                let _ = write!(out, "{name}(");
                self.fmt_expr(lhs, out, 0);
                if op.float_arity() == 2 {
                    out.push_str(", ");
                    self.fmt_expr(rhs, out, 0);
                }
                out.push(')');
            } else {
                self.fmt_expr(lhs, out, op.precedence());
                let _ = write!(out, " {} ", op.symbol());
                self.fmt_expr(rhs, out, op.precedence() + 1);
            }
        }
        Expr::Un { op, arg, .. } => {
            out.push_str(match op {
                crate::ir::UnOp::Neg => "-",
                crate::ir::UnOp::Not => "~",
                crate::ir::UnOp::LNot => "!",
            });
            self.fmt_expr(arg, out, 6);
        }
        Expr::Cond {
            op,
            lhs,
            rhs,
            bits,
        } => {
            // The width decides how the operands are read, and the output
            // has to say so: the same 32 bits are a negative number and a
            // large positive one, and a branch that asks "below" asks one
            // of them.
            let lhs = crate::ir::comparison_operand(lhs, *bits);
            let rhs = crate::ir::comparison_operand(rhs, *bits);
            if *bits < 8 {
                let ty = type_name(*bits, op.signed());
                // A constant is the value it is, whatever the width.
                if matches!(lhs, Expr::Const(_)) {
                    self.fmt_expr(lhs, out, 7);
                } else {
                    let _ = write!(out, "({ty})");
                    self.fmt_expr(lhs, out, 7);
                }
                let _ = write!(out, " {} ", op.symbol());
                if matches!(rhs, Expr::Const(_)) {
                    self.fmt_expr(rhs, out, 7);
                } else {
                    let _ = write!(out, "({ty})");
                    self.fmt_expr(rhs, out, 7);
                }
            } else {
                self.fmt_expr(lhs, out, 1);
                let _ = write!(out, " {} ", op.symbol());
                self.fmt_expr(rhs, out, 1);
            }
        }
        Expr::Bool { op, args } => {
            if *op == crate::ir::BoolOp::Not {
                out.push('!');
                self.fmt_expr(&args[0], out, 6);
            } else {
                self.fmt_expr(&args[0], out, op.precedence());
                let _ = write!(out, " {} ", op.symbol());
                self.fmt_expr(&args[1], out, op.precedence() + 1);
            }
        }
        Expr::FlagCond { cond, state, .. } => {
            // The condition on the flags is a question about the flags, and
            // the flags answer it with the values they hold.
            match state.condition(*cond) {
                Some(e) => self.fmt_expr(&e, out, min),
                None => {
                    let _ = write!(out, "<flags>");
                }
            }
        }
        Expr::Cast {
            kind,
            arg,
            from,
            to,
        } => {
            // Zero-extension to the machine width is the default reading of any
            // 32-bit value here, so it adds nothing.
            if *kind == crate::ir::CastKind::Zext && *to == 8 {
                self.fmt_expr(arg, out, min);
                if paren {
                    out.push(')');
                }
                return;
            }
            // A cast that does not change the width says nothing.
            if from == to {
                self.fmt_expr(arg, out, min);
                if paren {
                    out.push(')');
                }
                return;
            }
            let _ = write!(
                out,
                "({})",
                type_name(*to, *kind == crate::ir::CastKind::Sext)
            );
            self.fmt_expr(arg, out, 7);
        }
        Expr::Select { cond, yes, no } => {
            self.fmt_expr(cond, out, 1);
            out.push_str(" ? ");
            self.fmt_expr(yes, out, 0);
            out.push_str(" : ");
            self.fmt_expr(no, out, 0);
        }
        Expr::Phi(args) => {
            out.push_str("phi(");
            for (i, (bb, v)) in args.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                let _ = write!(out, "L{bb}: ");
                self.fmt_expr(v, out, 0);
            }
            out.push(')');
        }
        Expr::Intrinsic { op, args, bits } => {
            let _ = write!(out, "{}(", op.name_for(*bits));
            for (i, a) in args.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                self.fmt_expr(a, out, 0);
            }
            out.push(')');
        }
        Expr::Unknown(t) => {
            let _ = write!(out, "<{t}>");
        }
    }
    if paren {
        out.push(')');
    }
    }
}

/// Turn an SSA name into a C identifier: `rax.3` becomes `rax_3`.
pub fn name(ssa: &str) -> String {
    let mut out = String::with_capacity(ssa.len());
    let mut chars = ssa.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '.' => {
                out.push('_');
                // Skip the digits that follow and mark them with an underscore
                // only once.
                while chars.peek().is_some_and(|n| n.is_ascii_digit()) {
                    out.push(chars.next().unwrap());
                }
            }
            c if c.is_ascii_alphanumeric() || c == '_' => out.push(c),
            _ => out.push('_'),
        }
    }
    out
}

/// Render an expression, with SSA names and parameters turned into
/// identifiers.
pub fn expr(e: &Expr) -> String {
    let mut out = String::new();
    let r = Renderer {
        out: String::new(),
        indent: 0,
        aliases: Aliases { map: HashMap::new() },
        last_call: None,
    };
    r.fmt_expr(e, &mut out, 0);
    out
}

/// The register names that get renamed, and what they are renamed to.
///
/// The parameter registers are renamed to the parameter names the signature
/// uses, so the body reads `a0` where the signature says `a0`. The stack
/// pointer is renamed too: calling it `rsp` invites the reader to think the
/// value is still the live stack pointer, when it is the frame base.
struct Aliases {
    map: HashMap<String, String>,
}

impl Aliases {
    /// Build the renaming table for a decompiled function.
    fn new(d: &Decompiled) -> Self {
        let mut map = HashMap::new();
        for (reg, alias) in &d.param_aliases {
            map.insert(reg.clone(), alias.clone());
        }
        Aliases { map }
    }

    /// Identifier for a value: the parameter name for an incoming register, an
    /// SSA name otherwise.
    fn var_name(&self, v: &str) -> String {
        if let Some(alias) = self.map.get(v) {
            return alias.clone();
        }
        name(v)
    }
}

/// Identifier for a value, with no renaming table.
///
/// This is the x86-64 naming, kept for callers that render an expression
/// outside a decompiled function.
pub fn var_name(v: &str) -> String {
    for (reg, arg) in DEFAULT_PARAMS {
        if v == reg {
            return arg.to_string();
        }
    }
    name(v)
}

/// The x86-64 register names that read better under a descriptive name.
const DEFAULT_PARAMS: [(&str, &str); 7] = [
    ("rdi", "a0"),
    ("rsi", "a1"),
    ("rdx", "a2"),
    ("rcx", "a3"),
    ("r8", "a4"),
    ("r9", "a5"),
    // Not an argument: the stack pointer at entry, which is what a local
    // array's address is taken from. Calling it `rsp` invites the reader to
    // think the value is still the live stack pointer.
    ("rsp", "frame"),
];

/// Binding power of an expression, for parentheses.
fn prec(e: &Expr) -> (u8, bool) {
    match e {
        Expr::Bin { op, .. } => (op.precedence(), !op.is_float()),
        Expr::Cond { .. } => (0, true),
        Expr::Bool { op, .. } => (op.precedence(), true),
        Expr::FlagCond { .. } => (0, true),
        Expr::Un { .. } => (6, true),
        Expr::Cast { .. } => (7, true),
        Expr::Select { .. } => (0, true),
        _ => (9, false),
    }
}


/// Render a whole program: one function per entry, in the order given.
pub fn render_all(items: &[Decompiled]) -> String {
    let mut out = String::new();
    for d in items {
        out.push_str(&render(d));
        out.push('\n');
    }
    out
}

/// Count how many distinct locals appear, for summaries.
pub fn local_count(nodes: &[Node]) -> usize {
    let mut set: HashMap<String, ()> = HashMap::new();
    fn walk(nodes: &[Node], set: &mut HashMap<String, ()>) {
        for n in nodes {
            match n {
                Node::Assign { dest, .. } => {
                    if dest.starts_with("local_") || dest.starts_with("arg_") {
                        set.insert(dest.clone(), ());
                    }
                }
                Node::If {
                    then_body,
                    else_body,
                    ..
                } => {
                    walk(then_body, set);
                    walk(else_body, set);
                }
                Node::While { body, .. } | Node::DoWhile { body, .. } => walk(body, set),
                Node::Switch { cases, default, .. } => {
                    for (_, body) in cases {
                        walk(body, set);
                    }
                    walk(default, set);
                }
                _ => {}
            }
        }
    }
    walk(nodes, &mut set);
    set.len()
}
