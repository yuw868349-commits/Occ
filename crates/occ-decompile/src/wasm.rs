//! WebAssembly decompilation.
//!
//! WebAssembly is a stack machine whose control flow is structured, and both
//! facts change what a decompiler has to do here.
//!
//! The stack is why a WebAssembly body cannot be lifted the way a register
//! machine's is. There are no registers to read and write: the value an
//! instruction works on is the one the instructions before it left on the
//! stack, and a value pushed in one region and read in another has no name
//! until the region structure says which slot it sits in. Here the slots are
//! named, and the naming follows the nesting: a region begins at the stack
//! height it began at, so the value a branch carries out of it is the value
//! that was in that slot when the branch was taken.
//!
//! The structure is the second reason. A branch names a depth rather than an
//! address, and the depth counts the regions that are open at that point, so
//! the branch can only be read against the region stack. A `loop`'s label is
//! its *start* and a `block`'s label is its *end*, which is the difference
//! between repeating and leaving, and the reason the two are not the same
//! instruction.
//!
//! So this decompiler does not build a control-flow graph and hand it to the
//! generic structurer. The control flow is already structured in the byte
//! stream, and re-deriving a graph from it would lose the very thing the
//! module states: which branch continues a loop and which one leaves it.
//! The walk below follows the nesting directly and produces the structured
//! form the renderer prints.
//!
//! What the walk cannot express in C — a branch out of a region that is not
//! a loop — becomes a label and a `goto`, which is what the module means
//! and the only honest spelling of it.

use occ_core::{Addr, Function, Image, ImportKind, Instruction, WasmValueType};

use crate::Options;
use crate::ir::{BinOp, CastKind, CondOp, Decompiled, Expr, Intrinsic, Node};

/// Decompile one WebAssembly function.
///
/// The result is the pseudo-C the renderer prints: a signature with the
/// module's own types, and a body that follows the module's own nesting.
pub fn decompile(
    image: &Image,
    func: &Function,
    _opts: &Options,
    name_of: &dyn Fn(Addr) -> String,
) -> Decompiled {
    let sig = image
        .wasm_funcs
        .iter()
        .find(|s| s.body == Some(func.entry))
        .cloned();
    let params = sig.as_ref().map(|s| s.params as usize).unwrap_or(0);
    let results = sig.as_ref().map(|s| s.results as usize).unwrap_or(0);
    let ty = sig
        .as_ref()
        .and_then(|s| image.wasm_types.get(s.type_index as usize));
    let param_types: Vec<Option<String>> = match ty {
        Some(ty) => ty.params.iter().map(|p| Some(c_type(*p))).collect(),
        None => vec![None; params],
    };

    let mut walker = Walker::new(image, name_of, params, results);
    let mut body = walker.run(&func.insns);
    // The walk is faithful to the module's nesting; the reshape turns the
    // shapes that nesting leaves behind into the ones a reader expects.
    crate::reshape::reshape(&mut body);

    Decompiled {
        entry: func.entry,
        name: func.display_name(),
        params: (0..params).map(|i| format!("p{i}")).collect(),
        param_types,
        param_aliases: Vec::new(),
        ret_type: Some(ret_type(results, ty)),
        body,
        unstructured: 0,
        notes: walker.notes,
        source: None,
    }
}

/// The C type a WebAssembly value type is printed under.
///
/// The names are the ones the module uses, so a reader who knows the module
/// knows the output: an `i32` is an `i32` here as it is there.
fn c_type(ty: WasmValueType) -> String {
    match ty {
        WasmValueType::I32 => "i32".into(),
        WasmValueType::I64 => "i64".into(),
        WasmValueType::F32 => "float".into(),
        WasmValueType::F64 => "double".into(),
        WasmValueType::V128 => "v128".into(),
        WasmValueType::FuncRef => "funcref".into(),
        WasmValueType::ExternRef => "externref".into(),
    }
}

/// The type a function's result is printed under.
fn ret_type(results: usize, ty: Option<&occ_core::WasmFuncType>) -> String {
    if results == 0 {
        return "void".into();
    }
    match ty.and_then(|t| t.results.first()) {
        Some(first) => c_type(*first),
        None => "u32".into(),
    }
}

/// What a region is.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum FrameKind {
    /// The body of the function itself.
    Function,
    /// A `block`: its label is its end.
    Block,
    /// A `loop`: its label is its start.
    Loop,
    /// An `if`: its label is its end.
    If,
}

/// One region that is open.
struct Frame {
    /// What the region is.
    kind: FrameKind,
    /// The label a branch out of the region goes to, once one is needed.
    exit: Option<String>,
    /// The label at the region's start, for a loop.
    head: Option<String>,
    /// How many values the region carries when a branch leaves it.
    ///
    /// A `block` and an `if` carry their results out; a `loop` carries its
    /// parameters, which are the values a branch back to its start brings
    /// with it.
    arity: u8,
    /// The stack height when the region was opened.
    height: usize,
    /// The branches that go back to this region, when it is a loop.
    back_edges: Vec<BackEdge>,
    /// Whether a branch inside this region lands at the region's own end.
    ///
    /// This is the question that decides whether the code *after* the
    /// region runs. A branch that names the region's own depth leaves the
    /// region and continues after it; a branch that names a depth further
    /// out leaves the region without continuing after it.
    ///
    /// The flag cannot be a single "something escaped" bit, because a
    /// branch table names a different depth per case: one case may leave
    /// the region and another may continue after it, and the region is
    /// then reachable after all. Recording the landing per region keeps
    /// the two apart.
    landed: bool,
}

/// A branch that goes back to a loop's start.
struct BackEdge {
    /// Where in the loop's body the branch sits.
    at: usize,
    /// Whether the branch is conditional.
    conditional: bool,
    /// The condition of a conditional branch.
    cond: Option<Expr>,
}

/// The state of the decompiler while it walks one body.
struct Walker<'a> {
    /// The image the body belongs to.
    image: &'a Image,
    /// Resolves a call target to a name.
    name_of: &'a dyn Fn(Addr) -> String,
    /// The name of every imported function, by function index.
    imports: Vec<String>,
    /// How many parameters the function takes.
    params: u32,
    /// How many results the function returns.
    results: u32,
    /// The values on the operand stack.
    stack: Vec<Expr>,
    /// The regions that are open, outermost first.
    frames: Vec<Frame>,
    /// The region whose body is being filled.
    ///
    /// A branch that goes back to a loop is only the loop's condition when
    /// it sits at the loop's own level; one that sits inside a nested
    /// region is a jump to the loop's start, which is a different thing.
    body_frame: Option<usize>,
    /// The next label number.
    labels: usize,
    /// The next temporary number.
    temps: usize,
    /// Notes about what could not be lifted.
    notes: Vec<String>,
    /// Whether the path being walked has left the region it is in.
    terminated: bool,
    /// The call that was lifted last, and the stack depth it left.
    ///
    /// A call's result goes into a temporary, and the temporary is often
    /// copied straight into a local on the next instruction. When it is,
    /// the call can name the local itself and the copy disappears.
    last_call: Option<(String, usize)>,
}

impl<'a> Walker<'a> {
    /// A walker for one function.
    fn new(
        image: &'a Image,
        name_of: &'a dyn Fn(Addr) -> String,
        params: usize,
        results: usize,
    ) -> Self {
        // The imported functions come first in the index space, so the
        // n-th function import is the function of index n.
        let imports: Vec<String> = image
            .imports
            .iter()
            .filter(|i| i.kind == ImportKind::Normal)
            .map(|i| i.name.clone())
            .collect();
        Walker {
            image,
            name_of,
            imports,
            params: params as u32,
            results: results as u32,
            stack: Vec::new(),
            frames: Vec::new(),
            body_frame: None,
            labels: 0,
            temps: 0,
            notes: Vec::new(),
            terminated: false,
            last_call: None,
        }
    }

    /// Walk the whole body.
    fn run(&mut self, insns: &[Instruction]) -> Vec<Node> {
        self.frames.push(Frame {
            kind: FrameKind::Function,
            exit: None,
            head: None,
            arity: self.results as u8,
            height: 0,
            back_edges: Vec::new(),
            landed: false,
        });
        let mut pos = 0usize;
        let (mut items, fell) = self.parse_body(insns, &mut pos, false);
        if fell {
            // Falling off the end of a function returns what the module left
            // on the stack, which is what the module's own `end` means.
            let value = self.take_result();
            items.push(Node::Return(value));
        }
        items
    }

    /// A note about an approximation.
    fn note(&mut self, text: impl Into<String>) {
        let text = text.into();
        if !self.notes.contains(&text) {
            self.notes.push(text);
        }
    }

    /// A fresh label.
    fn label(&mut self) -> String {
        self.labels += 1;
        format!("L{}", self.labels)
    }

    /// A fresh temporary variable.
    fn temp(&mut self) -> String {
        self.temps += 1;
        format!("t{}", self.temps)
    }

    /// The name a local variable is printed under.
    ///
    /// The parameters are the first locals, and the module gives them
    /// names in the signature; naming them the same way in the body is what
    /// keeps the two in step.
    fn local_name(&self, index: u32) -> String {
        if index < self.params {
            format!("p{index}")
        } else {
            format!("local{index}")
        }
    }

    /// Pop a value off the stack.
    fn pop(&mut self) -> Expr {
        self.stack.pop().unwrap_or_else(|| {
            Expr::Unknown("the operand stack was empty where the module says a value is".into())
        })
    }

    /// The value the function returns, taken off the stack.
    fn take_result(&mut self) -> Option<Expr> {
        if self.results == 0 {
            return None;
        }
        let value = self.pop();
        if self.results > 1 {
            self.note(format!(
                "the function returns {} values; only the first is shown",
                self.results
            ));
        }
        Some(value)
    }

    /// Parse the instructions of a region body, up to its `else` or `end`.
    ///
    /// The `end` that closes the region is consumed; the `else` of an `if`
    /// is not, because the caller has to know which half it is looking at.
    ///
    /// The second value says whether the body fell through to its end: a
    /// body that ends in an unconditional branch does not, and the values
    /// it would have left on the stack are not there.
    fn parse_body(
        &mut self,
        insns: &[Instruction],
        pos: &mut usize,
        stop_at_else: bool,
    ) -> (Vec<Node>, bool) {
        let saved_frame = self.body_frame;
        let current = self.frames.len().saturating_sub(1);
        self.body_frame = Some(current);
        self.terminated = false;
        let mut items: Vec<Node> = Vec::new();
        let mut dead = false;
        while *pos < insns.len() {
            let insn = &insns[*pos];
            match insn.mnemonic.as_ref() {
                "end" => {
                    *pos += 1;
                    break;
                }
                "else" if stop_at_else => break,
                "block" | "loop" | "if" => {
                    *pos += 1;
                    let saved_stack = self.stack.clone();
                    let region = self.parse_region(insns, pos, insn);
                    if !dead {
                        // The region itself runs — the code reached it by
                        // falling into it. What the continuation decides is
                        // whether the code *after* it runs, which is a
                        // different question and one the region answers.
                        items.extend(region.items);
                        if let Some(label) = region.trailing {
                            items.push(Node::Label(label));
                        }
                        dead = !region.continuation;
                    }
                    if dead {
                        // The stack of unreachable code says nothing.
                        self.stack = saved_stack;
                    }
                    // The region's own last instruction says nothing about
                    // this body: what matters is whether the code after the
                    // region runs, which is the region's continuation.
                    self.terminated = dead;
                    continue;
                }
                _ => {}
            }
            *pos += 1;
            let before = items.len();
            self.parse_instruction(insn, &mut items);
            if dead {
                items.truncate(before);
            }
            if self.terminated {
                dead = true;
            }
        }
        self.body_frame = saved_frame;
        (items, !dead)
    }

    /// Parse one region: the `block`, `loop` or `if` and everything inside it.
    fn parse_region(
        &mut self,
        insns: &[Instruction],
        pos: &mut usize,
        opener: &Instruction,
    ) -> Region {
        let kind = match opener.mnemonic.as_ref() {
            "block" => FrameKind::Block,
            "loop" => FrameKind::Loop,
            _ => FrameKind::If,
        };
        let (params, results) = block_arity(&opener.text);
        let base = self.stack.clone();
        let height = base.len();

        self.frames.push(Frame {
            kind,
            exit: None,
            head: None,
            // A branch out of a region carries the region's results; a
            // branch back to a loop carries the loop's parameters, which
            // are the values the loop takes on entry.
            arity: if kind == FrameKind::Loop {
                params
            } else {
                results
            },
            height,
            back_edges: Vec::new(),
            landed: false,
        });

        // The condition of an `if` is on the stack when the `if` is read.
        let cond = if kind == FrameKind::If {
            Some(self.pop())
        } else {
            None
        };

        let (mut then_items, then_fell) = self.parse_body(insns, pos, kind == FrameKind::If);
        let then_stack = self.stack.clone();

        let mut else_items: Option<Vec<Node>> = None;
        let mut else_stack: Option<Vec<Expr>> = None;
        let mut else_fell = true;
        if kind == FrameKind::If && *pos < insns.len() && insns[*pos].mnemonic.as_ref() == "else" {
            *pos += 1;
            // The else half starts with the stack the `if` left, not with
            // the stack the then half built.
            self.stack = base.clone();
            let (items, fell) = self.parse_body(insns, pos, false);
            else_items = Some(items);
            else_fell = fell;
            else_stack = Some(self.stack.clone());
        }

        // What the region leaves on the stack.
        let mut out = base.clone();
        match kind {
            FrameKind::If => {
                if let Some(else_stack) = &else_stack {
                    // The two halves leave values that are the same only in
                    // the module's own accounting; in the output they are
                    // two expressions, so they are named and assigned on
                    // each side.
                    for i in 0..results as usize {
                        let name = self.temp();
                        let then_value = then_stack
                            .get(height + i)
                            .cloned()
                            .unwrap_or_else(|| unknown("a value the then half did not leave"));
                        let else_value = else_stack
                            .get(height + i)
                            .cloned()
                            .unwrap_or_else(|| unknown("a value the else half did not leave"));
                        if then_fell {
                            then_items.push(Node::Assign {
                                dest: name.clone(),
                                value: then_value,
                            });
                        }
                        if else_fell {
                            if let Some(list) = else_items.as_mut() {
                                list.push(Node::Assign {
                                    dest: name.clone(),
                                    value: else_value,
                                });
                            }
                        }
                        out.push(Expr::Var(name));
                    }
                } else {
                    for i in 0..results as usize {
                        out.push(
                            then_stack
                                .get(height + i)
                                .cloned()
                                .unwrap_or_else(|| unknown("a value the block did not leave")),
                        );
                    }
                }
            }
            FrameKind::Block | FrameKind::Loop => {
                // A region that fell through leaves its results; one that
                // ended in a branch leaves whatever the branch carried.
                let carried = if then_fell {
                    then_stack
                } else {
                    self.stack.clone()
                };
                for i in 0..results as usize {
                    out.push(
                        carried
                            .get(height + i)
                            .cloned()
                            .unwrap_or_else(|| unknown("a value the region did not leave")),
                    );
                }
            }
            FrameKind::Function => {}
        }
        self.stack = out;

        // The region is closed; its frame is gone.
        let frame = self.frames.pop().expect("the frame was pushed");
        let exit = frame.exit.clone();

        // Whether the code after the region runs.
        //
        // It runs when the region's body reaches the region's end, or when
        // a branch inside the region lands at that end — a branch to the
        // region's own depth continues after the region, while a branch to
        // a depth further out does not.
        //
        // A loop is different: a branch that names a loop goes back to its
        // *start*, so nothing lands at its end, and the code after the
        // loop runs only when the loop can be left by falling through it.
        let continuation = match kind {
            FrameKind::Block => then_fell || frame.landed,
            FrameKind::If => then_fell || else_fell || frame.landed,
            FrameKind::Loop => then_fell,
            FrameKind::Function => then_fell,
        };

        let items = match kind {
            FrameKind::If => {
                let else_body = else_items.unwrap_or_default();
                vec![Node::If {
                    cond: cond.unwrap_or_else(|| unknown("the condition of the branch")),
                    then_body: then_items,
                    else_body,
                }]
            }
            FrameKind::Block => then_items,
            FrameKind::Loop => self.loop_node(frame, then_items),
            FrameKind::Function => then_items,
        };

        Region {
            items,
            trailing: exit,
            continuation,
        }
    }

    /// The items a loop becomes.
    ///
    /// A loop with a branch back at the end of its body is a loop with a
    /// bottom test: the branch is its condition when it is conditional,
    /// and a loop that never ends when it is not. A branch back from
    /// anywhere else in the body runs the body again from the start,
    /// which is `continue`, and it is said that way rather than as a jump
    /// to a label the reader has to find. A loop with no branch at the
    /// end keeps a label and a `goto`, because its shape is a jump and
    /// not a loop the C syntax has a word for.
    fn loop_node(&mut self, frame: Frame, mut body: Vec<Node>) -> Vec<Node> {
        let back_edges = frame.back_edges;
        let head = frame.head;

        let trailing = back_edges.iter().position(|edge| edge.at + 1 == body.len());

        if let Some(trailing) = trailing {
            // The branch back to the loop is the last thing in it.
            body.pop();
            to_continue(&mut body, head.as_deref());
            // The label at the top is worth keeping only while something
            // still names it.
            let head = head.filter(|label| references(&body, label) > 0);
            let edge = &back_edges[trailing];
            let mut node = match (&edge.cond, edge.conditional) {
                (Some(cond), true) => Node::DoWhile {
                    body,
                    cond: cond.clone(),
                },
                _ => Node::While {
                    cond: Expr::Const(1),
                    body,
                },
            };
            if let Some(head) = head {
                match &mut node {
                    Node::DoWhile { body, .. } | Node::While { body, .. } => {
                        // A branch from outside the loop jumps to the top
                        // of the body, which is where the label goes.
                        body.insert(0, Node::Label(head));
                    }
                    _ => {}
                }
            }
            return vec![node];
        }

        // The general shape: a label at the top of the loop and a branch
        // back to it.
        let head = match head {
            Some(head) => head,
            None => self.label(),
        };
        for edge in &back_edges {
            if edge.at < body.len() {
                retarget(&mut body[edge.at], &head);
            }
        }
        let mut items = vec![Node::Label(head)];
        items.append(&mut body);
        items
    }

    /// Parse one instruction.
    fn parse_instruction(&mut self, insn: &Instruction, items: &mut Vec<Node>) {
        let mnemonic = insn.mnemonic.as_ref();
        let ops = operands(&insn.text);
        // Only the instruction right after a call can use its result.
        let last_call = self.last_call.take();
        match mnemonic {
            "unreachable" => {
                items.push(Node::Comment("the code here cannot be reached".to_string()));
                self.terminated = true;
            }
            "nop" => {}
            "return" => {
                let value = self.take_result();
                items.push(Node::Return(value));
                self.terminated = true;
            }
            "br" | "br_if" => {
                self.branch(mnemonic == "br_if", ops.first(), items);
            }
            "br_table" => {
                self.branch_table(&ops, items);
            }
            "call" | "return_call" => {
                self.call(mnemonic == "return_call", &ops, items);
            }
            "call_indirect" | "return_call_indirect" => {
                self.call_indirect(mnemonic == "return_call_indirect", &ops, items);
            }
            "drop" => {
                let _ = self.pop();
            }
            "select" | "select_t" => {
                let condition = self.pop();
                let no = self.pop();
                let yes = self.pop();
                self.stack.push(Expr::Select {
                    cond: Box::new(condition),
                    yes: Box::new(yes),
                    no: Box::new(no),
                });
            }
            "local.get" => {
                let index = number(ops.first().map(String::as_str).unwrap_or("")).unwrap_or(0);
                self.stack.push(Expr::Var(self.local_name(index as u32)));
            }
            "local.set" => {
                let index = number(ops.first().map(String::as_str).unwrap_or("")).unwrap_or(0);
                let name = self.local_name(index as u32);
                let depth = self.stack.len();
                let value = self.pop();
                if self.fold_call_into(&name, &value, depth, &last_call, items) {
                    // The call names the local itself; the copy is gone.
                } else if !is_self_assign(&name, &value) {
                    // A local written with the value it already holds is not
                    // an assignment; the compiler emits one whenever a value
                    // has to be both read and written, and printing it says
                    // nothing.
                    items.push(Node::Assign { dest: name, value });
                }
            }
            "local.tee" => {
                let index = number(ops.first().map(String::as_str).unwrap_or("")).unwrap_or(0);
                let name = self.local_name(index as u32);
                let depth = self.stack.len();
                let value = self.pop();
                let folded = self.fold_call_into(&name, &value, depth, &last_call, items);
                if !folded && !is_self_assign(&name, &value) {
                    items.push(Node::Assign {
                        dest: name.clone(),
                        value: value.clone(),
                    });
                }
                self.stack.push(Expr::Var(name));
            }
            "global.get" => {
                let index = number(ops.first().map(String::as_str).unwrap_or("")).unwrap_or(0);
                self.stack.push(Expr::Var(format!("global{index}")));
            }
            "global.set" => {
                let index = number(ops.first().map(String::as_str).unwrap_or("")).unwrap_or(0);
                let value = self.pop();
                items.push(Node::Assign {
                    dest: format!("global{index}"),
                    value,
                });
            }
            "i32.const" => {
                let value = number(ops.first().map(String::as_str).unwrap_or("")).unwrap_or(0);
                // A thirty-two bit constant is a thirty-two bit value: the
                // machine holds it in four bytes, and printing the sign
                // extension of it would say it is wider than it is.
                self.stack.push(Expr::Const(value as u32 as u64));
            }
            "i64.const" => {
                let value = number(ops.first().map(String::as_str).unwrap_or("")).unwrap_or(0);
                self.stack.push(Expr::Const(value as u64));
            }
            "f32.const" | "f64.const" => {
                let text = ops.first().map(String::as_str).unwrap_or("0");
                let bits = float_bits(text, mnemonic == "f32.const");
                self.stack.push(Expr::Const(bits));
            }
            "memory.size" => {
                self.stack
                    .push(Expr::Unknown("the size of the memory".into()));
            }
            "memory.grow" => {
                let _ = self.pop();
                self.stack
                    .push(Expr::Unknown("the size the memory grew to".into()));
            }
            "memory.fill" | "memory.copy" | "memory.init" => {
                let count = self.pop();
                let _ = self.pop();
                if mnemonic == "memory.copy" {
                    let _ = self.pop();
                }
                items.push(Node::Comment(format!("{mnemonic} of {count} bytes")));
            }
            "data.drop" | "elem.drop" => {
                items.push(Node::Comment(mnemonic.to_string()));
            }
            "table.init" | "table.copy" | "table.fill" => {
                let _ = self.pop();
                let _ = self.pop();
                let _ = self.pop();
                items.push(Node::Comment(mnemonic.to_string()));
            }
            "table.size" => {
                self.stack
                    .push(Expr::Unknown("the size of the table".into()));
            }
            "table.grow" => {
                let _ = self.pop();
                let _ = self.pop();
                self.stack
                    .push(Expr::Unknown("the size the table grew to".into()));
            }
            "table.get" => {
                let _ = self.pop();
                self.stack
                    .push(Expr::Unknown("the entry the table holds".into()));
            }
            "table.set" => {
                let _ = self.pop();
                let _ = self.pop();
                items.push(Node::Comment("table.set".into()));
            }
            "ref.null" => {
                self.stack.push(Expr::Unknown("a null reference".into()));
            }
            "ref.func" => {
                self.stack
                    .push(Expr::Unknown("a function reference".into()));
            }
            "ref.is_null" => {
                let value = self.pop();
                self.stack.push(Expr::Unknown(format!(
                    "whether {value} is a null reference"
                )));
            }
            _ => {
                if let Some(effect) = self.arithmetic(&ops, mnemonic) {
                    match effect {
                        Effect::Value(e) => self.stack.push(e),
                        Effect::Stmt(s) => items.push(s),
                    }
                    return;
                }
                self.note(format!("{mnemonic} is not lifted"));
                items.push(Node::Comment(insn.text.clone()));
            }
        }
    }

    /// Point a call's result at a local, when the call's result is what the
    /// local is written with.
    ///
    /// Returns whether the copy was folded away.
    fn fold_call_into(
        &mut self,
        name: &str,
        value: &Expr,
        depth: usize,
        last_call: &Option<(String, usize)>,
        items: &mut [Node],
    ) -> bool {
        let Some((temp, at_depth)) = last_call else {
            return false;
        };
        if *at_depth != depth {
            // Something else is on top of the call's result.
            return false;
        }
        if !matches!(value, Expr::Var(v) if *v == *temp) {
            return false;
        }
        let Some(Node::Call { ret: Some(r), .. }) = items.last_mut() else {
            return false;
        };
        if *r != *temp {
            return false;
        }
        *r = name.to_string();
        true
    }

    /// The arithmetic, the comparisons and the memory accesses.
    fn arithmetic(&mut self, ops: &[String], mnemonic: &str) -> Option<Effect> {
        // The memory accesses carry their width in the name.
        if let Some((width, store, signed)) = memory_access(mnemonic) {
            let offset = memarg(ops, "offset");
            if store {
                let value = self.pop();
                let address = self.pop();
                let address = add_offset(address, offset);
                return Some(Effect::Stmt(Node::Store {
                    addr: address,
                    value,
                    size: width,
                }));
            }
            let address = self.pop();
            let address = add_offset(address, offset);
            return Some(Effect::Value(Expr::Load {
                addr: Box::new(address),
                size: width,
                signed,
            }));
        }
        if let Some((op, width)) = integer_op(mnemonic) {
            let rhs = self.pop();
            let lhs = self.pop();
            return Some(Effect::Value(match op {
                Op::Bin(op) => Expr::Bin {
                    op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                    bits: width,
                },
                Op::Intrinsic(op) => Expr::Intrinsic {
                    op,
                    args: vec![lhs, rhs],
                    bits: width,
                },
            }));
        }
        if let Some((op, width)) = integer_cmp(mnemonic) {
            let rhs = self.pop();
            let lhs = self.pop();
            return Some(Effect::Value(Expr::Cond {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
                bits: width,
            }));
        }
        if let Some((op, width)) = float_op(mnemonic) {
            let rhs = self.pop();
            let lhs = self.pop();
            return Some(Effect::Value(match op {
                Op::Bin(op) => Expr::Bin {
                    op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                    bits: width,
                },
                Op::Intrinsic(op) => Expr::Intrinsic {
                    op,
                    args: vec![lhs, rhs],
                    bits: width,
                },
            }));
        }
        if let Some((op, width)) = float_cmp(mnemonic) {
            let rhs = self.pop();
            let lhs = self.pop();
            return Some(Effect::Value(Expr::Bin {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
                bits: width,
            }));
        }
        self.conversion(mnemonic)
    }

    /// The conversions between the number types.
    fn conversion(&mut self, mnemonic: &str) -> Option<Effect> {
        let unary = |op: BinOp, bits: u8, arg: Expr| {
            Effect::Value(Expr::Bin {
                op,
                lhs: Box::new(arg),
                rhs: Box::new(Expr::Const(0)),
                bits,
            })
        };
        match mnemonic {
            "i32.eqz" => {
                let value = self.pop();
                Some(Effect::Value(Expr::Cond {
                    op: CondOp::Eq,
                    lhs: Box::new(value),
                    rhs: Box::new(Expr::Const(0)),
                    bits: 4,
                }))
            }
            "i64.eqz" => {
                let value = self.pop();
                Some(Effect::Value(Expr::Cond {
                    op: CondOp::Eq,
                    lhs: Box::new(value),
                    rhs: Box::new(Expr::Const(0)),
                    bits: 8,
                }))
            }
            "i32.clz" | "i64.clz" => {
                let value = self.pop();
                let bits = if mnemonic.starts_with("i64") { 8 } else { 4 };
                Some(Effect::Value(Expr::Intrinsic {
                    op: Intrinsic::Lzcnt,
                    args: vec![value],
                    bits,
                }))
            }
            "i32.ctz" | "i64.ctz" => {
                let value = self.pop();
                let bits = if mnemonic.starts_with("i64") { 8 } else { 4 };
                Some(Effect::Value(Expr::Intrinsic {
                    op: Intrinsic::Tzcnt,
                    args: vec![value],
                    bits,
                }))
            }
            "i32.popcnt" | "i64.popcnt" => {
                let value = self.pop();
                let bits = if mnemonic.starts_with("i64") { 8 } else { 4 };
                Some(Effect::Value(Expr::Intrinsic {
                    op: Intrinsic::Popcnt,
                    args: vec![value],
                    bits,
                }))
            }
            "i32.extend8_s" | "i32.extend16_s" | "i64.extend8_s" | "i64.extend16_s"
            | "i64.extend32_s" => {
                let value = self.pop();
                let (from, to) = match mnemonic {
                    "i32.extend8_s" => (1u8, 4u8),
                    "i32.extend16_s" => (2, 4),
                    "i64.extend8_s" => (1, 8),
                    "i64.extend16_s" => (2, 8),
                    _ => (4, 8),
                };
                Some(Effect::Value(Expr::Cast {
                    kind: CastKind::Sext,
                    arg: Box::new(value),
                    from,
                    to,
                }))
            }
            "i32.wrap_i64" => {
                let value = self.pop();
                Some(Effect::Value(Expr::Cast {
                    kind: CastKind::Trunc,
                    arg: Box::new(value),
                    from: 8,
                    to: 4,
                }))
            }
            "i64.extend_i32_s" | "i64.extend_i32_u" => {
                let value = self.pop();
                Some(Effect::Value(Expr::Cast {
                    kind: if mnemonic.ends_with("_s") {
                        CastKind::Sext
                    } else {
                        CastKind::Zext
                    },
                    arg: Box::new(value),
                    from: 4,
                    to: 8,
                }))
            }
            "i32.reinterpret_f32"
            | "i64.reinterpret_f64"
            | "f32.reinterpret_i32"
            | "f64.reinterpret_i64" => {
                // The bits are the same either way; only the reading of
                // them changes.
                let value = self.pop();
                Some(Effect::Value(value))
            }
            "i32.trunc_f32_s" | "i32.trunc_f64_s" | "i64.trunc_f32_s" | "i64.trunc_f64_s"
            | "i32.trunc_f32_u" | "i32.trunc_f64_u" | "i64.trunc_f32_u" | "i64.trunc_f64_u" => {
                let value = self.pop();
                let source_bits = if mnemonic.contains("f32") { 4u8 } else { 8u8 };
                let op = match (mnemonic.starts_with("i64"), mnemonic.ends_with("_u")) {
                    (true, true) => BinOp::F2U64,
                    (true, false) => BinOp::F2I64,
                    (false, true) => BinOp::F2U32,
                    (false, false) => BinOp::F2I32,
                };
                Some(Effect::Value(Expr::Bin {
                    op,
                    lhs: Box::new(value),
                    rhs: Box::new(Expr::Const(0)),
                    bits: source_bits,
                }))
            }
            "i32.trunc_sat_f32_s"
            | "i32.trunc_sat_f64_s"
            | "i32.trunc_sat_f32_u"
            | "i32.trunc_sat_f64_u"
            | "i64.trunc_sat_f32_s"
            | "i64.trunc_sat_f64_s"
            | "i64.trunc_sat_f32_u"
            | "i64.trunc_sat_f64_u" => {
                let value = self.pop();
                Some(Effect::Value(saturating_truncation(mnemonic, value)))
            }
            "f32.convert_i32_s" | "f32.convert_i64_s" | "f64.convert_i32_s"
            | "f64.convert_i64_s" | "f32.convert_i32_u" | "f32.convert_i64_u"
            | "f64.convert_i32_u" | "f64.convert_i64_u" => {
                let value = self.pop();
                let single = mnemonic.starts_with("f32");
                let from_i32 = mnemonic.contains("_i32_");
                let unsigned = mnemonic.ends_with("_u");
                let op = if single { BinOp::I2F32 } else { BinOp::I2F64 };
                // A signed conversion of a narrow integer is the
                // sign-extended value; an unsigned one is the
                // zero-extended value. Saying so is what keeps the two
                // apart, and the extension is to the machine's own width
                // so that the conversion reads the number it is given.
                let arg = if from_i32 {
                    Expr::Cast {
                        kind: if unsigned {
                            CastKind::Zext
                        } else {
                            CastKind::Sext
                        },
                        arg: Box::new(value),
                        from: 4,
                        to: 8,
                    }
                } else {
                    value
                };
                Some(Effect::Value(Expr::Bin {
                    op,
                    lhs: Box::new(arg),
                    rhs: Box::new(Expr::Const(0)),
                    bits: 8,
                }))
            }
            "f64.demote_f32" | "f32.demote_f64" => {
                let value = self.pop();
                Some(unary(BinOp::F2F32, 4, value))
            }
            "f64.promote_f32" | "f32.promote_f64" => {
                let value = self.pop();
                Some(unary(BinOp::F2F64, 8, value))
            }
            "f32.abs" | "f64.abs" => {
                let value = self.pop();
                let bits = if mnemonic.starts_with("f32") { 4 } else { 8 };
                Some(unary(BinOp::FAbs, bits, value))
            }
            "f32.neg" | "f64.neg" => {
                let value = self.pop();
                let bits = if mnemonic.starts_with("f32") { 4 } else { 8 };
                Some(unary(BinOp::FNeg, bits, value))
            }
            "f32.sqrt" | "f64.sqrt" => {
                let value = self.pop();
                let bits = if mnemonic.starts_with("f32") { 4 } else { 8 };
                Some(unary(BinOp::FSqrt, bits, value))
            }
            "f32.ceil" | "f64.ceil" => {
                let value = self.pop();
                let bits = if mnemonic.starts_with("f32") { 4 } else { 8 };
                Some(unary(BinOp::FCeil, bits, value))
            }
            "f32.floor" | "f64.floor" => {
                let value = self.pop();
                let bits = if mnemonic.starts_with("f32") { 4 } else { 8 };
                Some(unary(BinOp::FFloor, bits, value))
            }
            "f32.nearest" | "f64.nearest" => {
                let value = self.pop();
                let bits = if mnemonic.starts_with("f32") { 4 } else { 8 };
                Some(unary(BinOp::FRound, bits, value))
            }
            _ => None,
        }
    }

    /// A branch, conditional or not.
    fn branch(&mut self, conditional: bool, depth: Option<&String>, items: &mut Vec<Node>) {
        let depth = number(depth.map(String::as_str).unwrap_or(""))
            .unwrap_or(0)
            .max(0) as usize;
        let Some(target) = self.frames.len().checked_sub(depth + 1) else {
            self.note(format!("a branch of depth {depth} names no open region"));
            items.push(Node::Comment(format!("a branch of depth {depth}")));
            self.terminated = true;
            return;
        };

        // The condition of a conditional branch is on top of the stack, and
        // it comes off before the values the branch carries. The value is
        // the condition: a branch on a non-zero value is a branch on the
        // value, which is what C's `if` already says.
        let cond = if conditional { Some(self.pop()) } else { None };

        // The values the branch carries are the ones on top of the stack.
        let arity = self.frames[target].arity as usize;
        let carried: Vec<Expr> = self
            .stack
            .iter()
            .skip(self.stack.len().saturating_sub(arity))
            .cloned()
            .collect();
        let height = self.frames[target].height;
        self.stack.truncate(height);
        self.stack.extend(carried);

        // A branch that names the region's own depth lands at that region's
        // end, so the code after the region runs. A branch to a depth
        // further out skips it. Which of the two it is decides whether the
        // region's continuation is reachable, and a branch table names a
        // different depth per case, so the landing is recorded per region
        // rather than as one "something escaped" bit.
        let node = match self.frames[target].kind {
            FrameKind::Function => {
                // A branch to the function's own label leaves the function,
                // and it carries the results with it.
                let value = self.take_result();
                Node::Return(value)
            }
            FrameKind::Loop => {
                // A branch to a loop goes back to its start. When it sits
                // at the loop's own level it is the loop's condition, and
                // where it goes is decided when the loop's shape is known;
                // when it sits deeper it is a jump to the top, and the
                // loop needs a label there.
                let at = items.len();
                if self.body_frame == Some(target) {
                    self.frames[target].back_edges.push(BackEdge {
                        at,
                        conditional,
                        cond: cond.clone(),
                    });
                    Node::Goto(String::new())
                } else {
                    Node::Goto(self.loop_head(target))
                }
            }
            FrameKind::Block | FrameKind::If => {
                self.frames[target].landed = true;
                Node::Goto(self.region_exit(target))
            }
        };

        match cond {
            Some(cond) => items.push(Node::If {
                cond,
                then_body: vec![node],
                else_body: Vec::new(),
            }),
            None => {
                items.push(node);
                self.terminated = true;
            }
        }
    }

    /// The label at the start of a loop, made on first use.
    fn loop_head(&mut self, target: usize) -> String {
        if let Some(head) = &self.frames[target].head {
            return head.clone();
        }
        let head = self.label();
        self.frames[target].head = Some(head.clone());
        head
    }

    /// The label at the end of a region, made on first use.
    fn region_exit(&mut self, target: usize) -> String {
        if let Some(exit) = &self.frames[target].exit {
            return exit.clone();
        }
        let exit = self.label();
        self.frames[target].exit = Some(exit.clone());
        exit
    }

    /// A branch table: a dispatch over the index it pops.
    fn branch_table(&mut self, ops: &[String], items: &mut Vec<Node>) {
        // The labels are written after the word "labels", and the default
        // after the word "default".
        let mut labels: Vec<u32> = Vec::new();
        let mut default: Option<u32> = None;
        let mut in_labels = false;
        for op in ops {
            if op == "labels" {
                in_labels = true;
                continue;
            }
            if op == "default" {
                in_labels = false;
                continue;
            }
            if in_labels {
                if let Some(depth) = number(op) {
                    labels.push(depth as u32);
                }
            } else if let Some(depth) = number(op) {
                default = Some(depth as u32);
            }
        }
        let index = self.pop();
        let saved = self.stack.clone();
        let mut cases: Vec<(u64, Vec<Node>)> = Vec::new();
        for (i, depth) in labels.iter().enumerate() {
            self.stack = saved.clone();
            let node = self.branch_node(*depth as usize);
            cases.push((i as u64, vec![node]));
        }
        self.stack = saved.clone();
        let node = self.branch_node(default.unwrap_or(0) as usize);
        items.push(Node::Switch {
            expr: index,
            cases,
            default: vec![node],
        });
        self.terminated = true;
    }

    /// The node a branch to a region of the given depth becomes.
    fn branch_node(&mut self, depth: usize) -> Node {
        let mut scratch: Vec<Node> = Vec::new();
        self.branch(false, Some(&depth.to_string()), &mut scratch);
        // The branch pushed exactly one node; anything else it pushed is
        // not part of the case.
        if scratch.len() == 1 {
            scratch.pop().expect("the branch pushed a node")
        } else {
            Node::Comment(format!("a branch of depth {depth}"))
        }
    }

    /// A call to a function the module names.
    fn call(&mut self, tail: bool, ops: &[String], items: &mut Vec<Node>) {
        let index = number(ops.first().map(String::as_str).unwrap_or("")).unwrap_or(0) as u32;
        let target = ops
            .iter()
            .skip_while(|o| *o != "->")
            .nth(1)
            .and_then(|o| o.strip_prefix("0x"))
            .and_then(|o| u64::from_str_radix(o, 16).ok())
            .map(Addr);
        let sig = self.image.wasm_funcs.get(index as usize);
        let params = sig.map(|s| s.params as usize).unwrap_or(0);
        let results = sig.map(|s| s.results as usize).unwrap_or(0);
        let mut args = Vec::with_capacity(params);
        for _ in 0..params {
            args.insert(0, self.pop());
        }
        let name = match target {
            Some(addr) => (self.name_of)(addr),
            None => self
                .imports
                .get(index as usize)
                .cloned()
                .unwrap_or_else(|| format!("function {index}")),
        };
        let ret = if results > 0 { Some(self.temp()) } else { None };
        items.push(Node::Call {
            name,
            args,
            ret: ret.clone(),
        });
        if let Some(ret) = ret {
            self.stack.push(Expr::Var(ret.clone()));
            self.last_call = Some((ret, self.stack.len()));
        }
        if tail {
            // A tail call returns what the callee returns.
            let value = self.take_result();
            items.push(Node::Return(value));
            self.terminated = true;
        }
    }

    /// A call through a table.
    fn call_indirect(&mut self, tail: bool, ops: &[String], items: &mut Vec<Node>) {
        let ty = ops
            .iter()
            .position(|o| o == "type")
            .and_then(|i| ops.get(i + 1))
            .and_then(|o| number(o))
            .unwrap_or(0) as usize;
        let table = ops
            .iter()
            .position(|o| o == "table")
            .and_then(|i| ops.get(i + 1))
            .and_then(|o| number(o))
            .unwrap_or(0);
        let sig = self.image.wasm_types.get(ty);
        let params = sig.map(|t| t.params.len()).unwrap_or(0);
        let results = sig.map(|t| t.results.len()).unwrap_or(0);
        // The index into the table is on top of the stack, above the
        // arguments.
        let index = self.pop();
        let mut args = Vec::with_capacity(params);
        for _ in 0..params {
            args.insert(0, self.pop());
        }
        let name = format!("table{table}[{}]", crate::render::expr(&index));
        let ret = if results > 0 { Some(self.temp()) } else { None };
        items.push(Node::Call {
            name,
            args,
            ret: ret.clone(),
        });
        if let Some(ret) = ret {
            self.stack.push(Expr::Var(ret.clone()));
            self.last_call = Some((ret, self.stack.len()));
        }
        if tail {
            let value = self.take_result();
            items.push(Node::Return(value));
            self.terminated = true;
        }
    }
}

/// What one region contributes to the body that holds it.
struct Region {
    /// The items the region contributes.
    items: Vec<Node>,
    /// A label to place after them, for a region that is branched out of.
    trailing: Option<String>,
    /// Whether the code after the region runs.
    ///
    /// A region that is left only by branches that skip what follows it
    /// makes the rest of the body unreachable, and the body after it must
    /// not be shown: showing it would print code that cannot run.
    continuation: bool,
}

/// What an instruction produced.
enum Effect {
    /// A value for the operand stack.
    Value(Expr),
    /// A statement for the body.
    Stmt(Node),
}

/// The operation an instruction performs.
enum Op {
    /// An arithmetic or bitwise operation.
    Bin(BinOp),
    /// An operation the machine word has no arithmetic for.
    Intrinsic(Intrinsic),
}

/// Point a branch at a label, wherever in the statement it sits.
fn retarget(node: &mut Node, head: &str) {
    match node {
        Node::Goto(_) => *node = Node::Goto(head.to_string()),
        Node::If { then_body, .. } => {
            for n in then_body {
                retarget(n, head);
            }
        }
        _ => {}
    }
}

/// Say a branch back to the top of a loop as `continue`.
///
/// The branch and the keyword mean the same thing: the body runs again
/// from its first statement. A branch inside a nested loop is left alone,
/// because there `continue` names the inner loop and the branch names the
/// outer one, and saying it as the keyword would move it.
fn to_continue(body: &mut [Node], head: Option<&str>) {
    for node in body.iter_mut() {
        match node {
            Node::Goto(label) if label.is_empty() || Some(label.as_str()) == head => {
                *node = Node::Continue;
            }
            Node::If {
                then_body,
                else_body,
                ..
            } => {
                to_continue(then_body, head);
                to_continue(else_body, head);
            }
            _ => {}
        }
    }
}

/// How many branches in a body name a label.
fn references(body: &[Node], label: &str) -> usize {
    let mut count = 0;
    for node in body {
        match node {
            Node::Goto(name) if name == label => count += 1,
            Node::If {
                then_body,
                else_body,
                ..
            } => {
                count += references(then_body, label);
                count += references(else_body, label);
            }
            Node::While { body, .. } | Node::DoWhile { body, .. } => {
                count += references(body, label);
            }
            Node::Switch { cases, default, .. } => {
                for (_, case) in cases {
                    count += references(case, label);
                }
                count += references(default, label);
            }
            _ => {}
        }
    }
    count
}

/// A value the module does not state.
fn unknown(text: &str) -> Expr {
    Expr::Unknown(text.to_string())
}

/// Whether an assignment says that a variable holds what it already holds.
fn is_self_assign(dest: &str, value: &Expr) -> bool {
    matches!(value, Expr::Var(v) if v == dest)
}

/// The parameters and the results a block carries, as the decoder printed them.
fn block_arity(text: &str) -> (u8, u8) {
    let rest = text
        .split_once(char::is_whitespace)
        .map(|(_, r)| r.trim())
        .unwrap_or("");
    let mut params = 0u8;
    let mut results = 0u8;
    for part in rest.split(',') {
        let part = part.trim().trim_start_matches('(').trim();
        if part.starts_with("param ") {
            params += 1;
        } else if part.starts_with("result ") {
            results += 1;
        }
    }
    (params, results)
}

/// The operands of an instruction, split on spaces and commas.
///
/// The mnemonic is not one of them: the text reads `local.get 0`, and the
/// operand of that instruction is the `0`.
fn operands(text: &str) -> Vec<String> {
    let mut out: Vec<String> = text
        .split_whitespace()
        .flat_map(|word| word.split(','))
        .map(|word| word.trim().to_string())
        .filter(|word| !word.is_empty())
        .collect();
    if !out.is_empty() {
        out.remove(0);
    }
    out
}

/// The number an operand states.
fn number(text: &str) -> Option<i64> {
    let text = text.trim();
    if let Some(hex) = text.strip_prefix("0x") {
        return i64::from_str_radix(hex, 16).ok();
    }
    text.parse::<i64>().ok()
}

/// The bits a floating point constant holds.
fn float_bits(text: &str, single: bool) -> u64 {
    let text = text.trim();
    let value = match text {
        "inf" | "+inf" => f64::INFINITY,
        "-inf" => f64::NEG_INFINITY,
        "nan" | "-nan" => f64::NAN,
        _ => text.parse::<f64>().unwrap_or(0.0),
    };
    if single {
        u64::from((value as f32).to_bits())
    } else {
        value.to_bits()
    }
}

/// The immediate of a memory access.
fn memarg(ops: &[String], name: &str) -> u64 {
    let prefix = format!("{name}=");
    for op in ops {
        if let Some(rest) = op.strip_prefix(&prefix) {
            return rest.parse::<u64>().unwrap_or(0);
        }
    }
    0
}

/// The address of a memory access, with its immediate added.
fn add_offset(address: Expr, offset: u64) -> Expr {
    if offset == 0 {
        return address;
    }
    Expr::Bin {
        op: BinOp::Add,
        lhs: Box::new(address),
        rhs: Box::new(Expr::Const(offset)),
        bits: 4,
    }
}

/// The width of a memory access, whether it writes, and whether it
/// sign-extends.
fn memory_access(name: &str) -> Option<(u8, bool, bool)> {
    let (body, _store) = if let Some(rest) = name.strip_prefix("i32.") {
        (rest, false)
    } else if let Some(rest) = name.strip_prefix("i64.") {
        (rest, false)
    } else if let Some(rest) = name.strip_prefix("f32.") {
        (rest, false)
    } else {
        let rest = name.strip_prefix("f64.")?;
        (rest, false)
    };
    if let Some(rest) = body.strip_prefix("store") {
        let width = match rest {
            "" => 4,
            "8" => 1,
            "16" => 2,
            "32" => 4,
            _ => return None,
        };
        return Some((width, true, false));
    }
    if let Some(rest) = body.strip_prefix("load") {
        let (width, signed) = match rest {
            "" => (4u8, false),
            "8_s" => (1, true),
            "8_u" => (1, false),
            "16_s" => (2, true),
            "16_u" => (2, false),
            "32_s" => (4, true),
            "32_u" => (4, false),
            _ => return None,
        };
        return Some((width, false, signed));
    }
    None
}

/// The operation an integer instruction performs.
fn integer_op(name: &str) -> Option<(Op, u8)> {
    let (kind, width) = if let Some(rest) = name.strip_prefix("i32.") {
        (rest, 4u8)
    } else {
        let rest = name.strip_prefix("i64.")?;
        (rest, 8u8)
    };
    let op = match kind {
        "add" => Op::Bin(BinOp::Add),
        "sub" => Op::Bin(BinOp::Sub),
        "mul" => Op::Bin(BinOp::Mul),
        "and" => Op::Bin(BinOp::And),
        "or" => Op::Bin(BinOp::Or),
        "xor" => Op::Bin(BinOp::Xor),
        "shl" => Op::Bin(BinOp::Shl),
        "shr_s" => Op::Bin(BinOp::Sar),
        "shr_u" => Op::Bin(BinOp::Shr),
        "div_s" => Op::Bin(BinOp::Div),
        "div_u" => Op::Bin(BinOp::UDiv),
        "rem_s" => Op::Bin(BinOp::Mod),
        "rem_u" => Op::Bin(BinOp::UMod),
        "rotl" => Op::Intrinsic(Intrinsic::Rol),
        "rotr" => Op::Intrinsic(Intrinsic::Ror),
        _ => return None,
    };
    Some((op, width))
}

/// The relation an integer comparison asks about.
fn integer_cmp(name: &str) -> Option<(CondOp, u8)> {
    let (kind, width) = if let Some(rest) = name.strip_prefix("i32.") {
        (rest, 4u8)
    } else {
        let rest = name.strip_prefix("i64.")?;
        (rest, 8u8)
    };
    let op = match kind {
        "eq" => CondOp::Eq,
        "ne" => CondOp::Ne,
        "lt_s" => CondOp::Slt,
        "lt_u" => CondOp::Ult,
        "gt_s" => CondOp::Sgt,
        "gt_u" => CondOp::Ugt,
        "le_s" => CondOp::Sle,
        "le_u" => CondOp::Ule,
        "ge_s" => CondOp::Sge,
        "ge_u" => CondOp::Uge,
        _ => return None,
    };
    Some((op, width))
}

/// The operation a floating point instruction performs.
fn float_op(name: &str) -> Option<(Op, u8)> {
    let (kind, width) = if let Some(rest) = name.strip_prefix("f32.") {
        (rest, 4u8)
    } else {
        let rest = name.strip_prefix("f64.")?;
        (rest, 8u8)
    };
    let op = match kind {
        "add" => Op::Bin(BinOp::FAdd),
        "sub" => Op::Bin(BinOp::FSub),
        "mul" => Op::Bin(BinOp::FMul),
        "div" => Op::Bin(BinOp::FDiv),
        "min" => Op::Bin(BinOp::FMin),
        "max" => Op::Bin(BinOp::FMax),
        "copysign" => Op::Intrinsic(Intrinsic::CopySign),
        _ => return None,
    };
    Some((op, width))
}

/// The relation a floating point comparison asks about.
fn float_cmp(name: &str) -> Option<(BinOp, u8)> {
    let (kind, width) = if let Some(rest) = name.strip_prefix("f32.") {
        (rest, 4u8)
    } else {
        let rest = name.strip_prefix("f64.")?;
        (rest, 8u8)
    };
    let op = match kind {
        "eq" => BinOp::FEq,
        "ne" => BinOp::FNe,
        "lt" => BinOp::FLt,
        "gt" => BinOp::FGt,
        "le" => BinOp::FLe,
        "ge" => BinOp::FGe,
        _ => return None,
    };
    Some((op, width))
}

/// The value a saturating truncation produces.
///
/// The saturation is a pair of comparisons and a choice, which is what it
/// is: a value below the range becomes the lowest one in it, a value above
/// it becomes the highest, and anything between is truncated.
fn saturating_truncation(mnemonic: &str, value: Expr) -> Expr {
    let unsigned = mnemonic.ends_with("_u");
    let wide = mnemonic.starts_with("i64");
    let source_bits = if mnemonic.contains("f32") { 4u8 } else { 8u8 };
    // The thresholds are the first values that do not fit, and every one
    // of them is a power of two, so every one of them is a float exactly.
    let (low, high, low_sat, high_sat, op) = match (unsigned, wide) {
        (false, false) => (
            -2147483648.0f64,
            2147483648.0,
            0x8000_0000u64,
            0x7fff_ffff,
            BinOp::F2I32,
        ),
        (false, true) => (
            -9223372036854775808.0,
            9223372036854775808.0,
            0x8000_0000_0000_0000,
            0x7fff_ffff_ffff_ffff,
            BinOp::F2I64,
        ),
        (true, false) => (0.0f64, 4294967296.0, 0u64, 0xffff_ffff, BinOp::F2U32),
        (true, true) => (0.0f64, 18446744073709551616.0, 0, u64::MAX, BinOp::F2U64),
    };
    let truncated = Expr::Bin {
        op,
        lhs: Box::new(value.clone()),
        rhs: Box::new(Expr::Const(0)),
        bits: source_bits,
    };
    Expr::Select {
        cond: Box::new(Expr::Bin {
            op: BinOp::FLt,
            lhs: Box::new(value.clone()),
            rhs: Box::new(Expr::Const(float_bits(&low.to_string(), source_bits == 4))),
            bits: source_bits,
        }),
        yes: Box::new(Expr::Const(low_sat)),
        no: Box::new(Expr::Select {
            cond: Box::new(Expr::Bin {
                op: BinOp::FGe,
                lhs: Box::new(value),
                rhs: Box::new(Expr::Const(float_bits(&high.to_string(), source_bits == 4))),
                bits: source_bits,
            }),
            yes: Box::new(Expr::Const(high_sat)),
            no: Box::new(truncated),
        }),
    }
}
