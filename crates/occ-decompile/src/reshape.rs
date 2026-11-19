//! The last pass over a structured body.
//!
//! A body that follows the module's own nesting exactly is faithful, and
//! it is not what a reader expects. A branch that skips over the rest of a
//! region is a `goto` and a label in that body, while the same code in C
//! is an `if` whose body is the code that was skipped. A branch that
//! leaves a loop is a `goto` to a label past the loop, while the same
//! code in C is a `break`.
//!
//! The passes below rewrite those shapes. Every one of them fires only
//! when it can show that the result means the same thing, and a shape it
//! cannot show is left exactly as it was: a body that keeps its `goto`
//! is honest, and a body that is rewritten into something else is not.

use std::collections::HashMap;

use crate::ir::{BoolOp, Expr, Node};

/// Rewrite a body into the shape the source most likely had.
///
/// The passes feed each other — a branch that becomes a `break` can leave
/// a label behind that another pass then removes — so they run until the
/// body stops changing. The bound is a safety net: each pass either
/// changes the body or it does not, and the body is finite.
pub fn reshape(body: &mut Vec<Node>) {
    for _ in 0..16 {
        if !pass(body) {
            break;
        }
    }
}

/// One round of the passes over one list of statements.
fn pass(body: &mut Vec<Node>) -> bool {
    let mut changed = false;
    // The shapes inside a list come first: a branch that leaves a region
    // is easier to see once the region is in one piece.
    for node in body.iter_mut() {
        match node {
            Node::If {
                then_body,
                else_body,
                ..
            } => {
                changed |= pass(then_body);
                changed |= pass(else_body);
            }
            Node::While { body, .. } | Node::DoWhile { body, .. } => changed |= pass(body),
            Node::Switch { cases, default, .. } => {
                for (_, case) in cases.iter_mut() {
                    changed |= pass(case);
                }
                changed |= pass(default);
            }
            _ => {}
        }
    }
    changed |= fold_break(body);
    changed |= fold_forward_goto(body);
    changed |= fold_flip(body);
    changed
}

/// Turn `if (c) goto L; rest` into `if (!c) { rest }`.
///
/// The branch skips the code between itself and the label, so the code is
/// the body of the branch taken when the condition does not hold. That is
/// the same program written the other way round, and it is the way round
/// a person writes it.
///
/// The rewrite only fires when the label is in the same list, because
/// that is what says the code between them is the code the branch skips.
/// A branch to another region's end leaves that region, and the code it
/// skips is not in this list at all.
fn fold_flip(body: &mut Vec<Node>) -> bool {
    let info = Info::of(body);
    for i in 0..body.len() {
        let Some(Node::If {
            cond,
            then_body,
            else_body,
        }) = body.get(i)
        else {
            continue;
        };
        // Only a branch has nothing on the other side. A conditional in
        // the module's own shape has two halves, and moving code into one
        // of them would drop the other.
        if !else_body.is_empty() {
            continue;
        }
        let [Node::Goto(target)] = then_body.as_slice() else {
            continue;
        };
        let target = target.clone();
        let Some(j) = (i + 1..body.len()).find(|k| match &body[*k] {
            Node::Label(l) => *l == target,
            _ => false,
        }) else {
            continue;
        };
        // Nothing between the branch and the label: the branch goes to
        // the next statement, and there is no code to move.
        if j == i + 1 {
            continue;
        }
        // The label goes away with the branch, so nothing else may name it.
        if info.refs.get(&target).copied().unwrap_or(0) != 1 {
            continue;
        }
        // A label inside the moved code that something outside jumps to
        // would end up inside a block, and the jump would land in the
        // middle of it.
        if !closed(body, i + 1, j, &info) {
            continue;
        }

        let cond = negate(cond);
        let region: Vec<Node> = body.drain(i + 1..j).collect();
        // The label came back with the drained code's place; it is the
        // statement that now follows the branch.
        body.remove(i + 1);
        body[i] = Node::If {
            cond,
            then_body: region,
            else_body: Vec::new(),
        };
        return true;
    }
    false
}

/// Turn a branch that leaves a loop into a `break`.
///
/// A branch whose label is the statement right after the loop leaves the
/// loop and lands on the next statement, which is what `break` says. The
/// label has to be the very next statement: a branch to a label further
/// down skips the code in between, and `break` does not skip anything.
fn fold_break(body: &mut Vec<Node>) -> bool {
    for i in 0..body.len() {
        let Some(Node::Label(label)) = body.get(i + 1) else {
            continue;
        };
        let label = label.clone();
        let mut hits = 0;
        {
            let (Some(Node::While {
                body: loop_body, ..
            })
            | Some(Node::DoWhile {
                body: loop_body, ..
            })) = body.get_mut(i)
            else {
                continue;
            };
            break_out(loop_body, &label, &mut hits);
        }
        if hits == 0 {
            continue;
        }
        // The label is gone when nothing names it any more.
        let info = Info::of(body);
        if info.refs.get(&label).copied().unwrap_or(0) == 0 {
            body.remove(i + 1);
        }
        return true;
    }
    false
}

/// Replace the branches to a label with `break`, inside one loop.
///
/// A branch inside a loop nested in this one leaves that inner loop as
/// well, and `break` would only leave the inner one, so the inner loops
/// are left alone.
fn break_out(body: &mut [Node], label: &str, hits: &mut usize) {
    for node in body.iter_mut() {
        match node {
            Node::Goto(l) if l == label => {
                *node = Node::Break;
                *hits += 1;
            }
            Node::If {
                then_body,
                else_body,
                ..
            } => {
                break_out(then_body, label, hits);
                break_out(else_body, label, hits);
            }
            Node::Switch { cases, default, .. } => {
                for (_, case) in cases.iter_mut() {
                    break_out(case, label, hits);
                }
                break_out(default, label, hits);
            }
            // A loop inside the loop has its own `break`.
            Node::While { .. } | Node::DoWhile { .. } => {}
            _ => {}
        }
    }
}

/// Remove a branch that goes to the statement right after it.
///
/// A `goto` to the next statement says nothing, and the label it names
/// says nothing either once the branch is gone.
fn fold_forward_goto(body: &mut Vec<Node>) -> bool {
    let mut changed = false;
    let mut i = 0;
    while i + 1 < body.len() {
        let target = match &body[i] {
            Node::Goto(l) => l.clone(),
            _ => {
                i += 1;
                continue;
            }
        };
        let is_label = matches!(&body[i + 1], Node::Label(l) if *l == target);
        if !is_label {
            i += 1;
            continue;
        }
        let info = Info::of(body);
        // The label goes away, so nothing else may name it.
        if info.refs.get(&target).copied().unwrap_or(0) != 1 {
            i += 1;
            continue;
        }
        body.drain(i..i + 2);
        changed = true;
    }
    changed
}

/// Whether the code in a range is closed under its own labels.
///
/// A label defined in the range that something outside the range jumps to
/// would move into a block, and the jump would land in the middle of it.
fn closed(body: &[Node], from: usize, to: usize, info: &Info) -> bool {
    let mut labels = Vec::new();
    collect_labels(&body[from..to], &mut labels);
    for label in labels {
        // A label defined twice is two different places, and a branch
        // names the nearest one.
        if info.defs.get(&label).copied().unwrap_or(0) != 1 {
            return false;
        }
        let inside = count_gotos(&body[from..to], &label);
        if inside != info.refs.get(&label).copied().unwrap_or(0) {
            return false;
        }
    }
    true
}

/// The labels a range of statements defines.
fn collect_labels(body: &[Node], out: &mut Vec<String>) {
    for node in body {
        match node {
            Node::Label(l) => out.push(l.clone()),
            Node::If {
                then_body,
                else_body,
                ..
            } => {
                collect_labels(then_body, out);
                collect_labels(else_body, out);
            }
            Node::While { body, .. } | Node::DoWhile { body, .. } => collect_labels(body, out),
            Node::Switch { cases, default, .. } => {
                for (_, case) in cases {
                    collect_labels(case, out);
                }
                collect_labels(default, out);
            }
            _ => {}
        }
    }
}

/// How many branches in a range of statements name a label.
fn count_gotos(body: &[Node], label: &str) -> usize {
    let mut total = 0;
    for node in body {
        match node {
            Node::Goto(l) if l == label => total += 1,
            Node::If {
                then_body,
                else_body,
                ..
            } => {
                total += count_gotos(then_body, label);
                total += count_gotos(else_body, label);
            }
            Node::While { body, .. } | Node::DoWhile { body, .. } => {
                total += count_gotos(body, label);
            }
            Node::Switch { cases, default, .. } => {
                for (_, case) in cases {
                    total += count_gotos(case, label);
                }
                total += count_gotos(default, label);
            }
            _ => {}
        }
    }
    total
}

/// The condition that holds when the one given does not.
fn negate(e: &Expr) -> Expr {
    match e {
        // A comparison reads best with the comparison reversed: `a >= b`
        // becomes `a < b`, not `!(a >= b)`.
        Expr::Cond { op, lhs, rhs, bits } => Expr::Cond {
            op: op.negated(),
            lhs: lhs.clone(),
            rhs: rhs.clone(),
            bits: *bits,
        },
        // A negated condition is already the condition that holds when
        // this one does not.
        Expr::Bool {
            op: BoolOp::Not,
            args,
        } => args[0].clone(),
        // Anything else is a value, and a value is true when it is not
        // zero.
        _ => Expr::Bool {
            op: BoolOp::Eq,
            args: vec![e.clone(), Expr::Const(0)],
        },
    }
}

/// What a body says about its labels.
struct Info {
    /// How many branches name each label.
    refs: HashMap<String, usize>,
    /// How many times each label is defined.
    defs: HashMap<String, usize>,
}

impl Info {
    /// Read the labels out of a body.
    fn of(body: &[Node]) -> Info {
        let mut info = Info {
            refs: HashMap::new(),
            defs: HashMap::new(),
        };
        info.scan(body);
        info
    }

    /// Read one list of statements.
    fn scan(&mut self, body: &[Node]) {
        for node in body {
            match node {
                Node::Goto(l) => *self.refs.entry(l.clone()).or_insert(0) += 1,
                Node::Label(l) => *self.defs.entry(l.clone()).or_insert(0) += 1,
                Node::If {
                    then_body,
                    else_body,
                    ..
                } => {
                    self.scan(then_body);
                    self.scan(else_body);
                }
                Node::While { body, .. } | Node::DoWhile { body, .. } => self.scan(body),
                Node::Switch { cases, default, .. } => {
                    for (_, case) in cases {
                        self.scan(case);
                    }
                    self.scan(default);
                }
                _ => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::CondOp;

    /// A body of statements, for the tests.
    fn body(items: Vec<Node>) -> Vec<Node> {
        items
    }

    /// A branch on a condition to a label.
    fn branch(cond: Expr, label: &str) -> Node {
        Node::If {
            cond,
            then_body: vec![Node::Goto(label.to_string())],
            else_body: Vec::new(),
        }
    }

    /// A comparison, for the conditions.
    fn compare(op: CondOp) -> Expr {
        Expr::Cond {
            op,
            lhs: Box::new(Expr::Var("a".into())),
            rhs: Box::new(Expr::Const(0)),
            bits: 32,
        }
    }

    #[test]
    fn a_branch_that_skips_code_becomes_an_if() {
        let mut items = body(vec![
            branch(compare(CondOp::Sge), "L1"),
            Node::Assign {
                dest: "x".into(),
                value: Expr::Const(1),
            },
            Node::Label("L1".into()),
        ]);
        reshape(&mut items);
        let Node::If {
            cond,
            then_body,
            else_body,
        } = &items[0]
        else {
            panic!("the branch became a conditional: {items:?}");
        };
        assert!(else_body.is_empty());
        assert_eq!(then_body.len(), 1, "the skipped code is the body");
        assert!(
            matches!(
                &cond,
                Expr::Cond {
                    op: CondOp::Slt,
                    ..
                }
            ),
            "the condition is reversed: {cond:?}"
        );
        assert!(
            !items.iter().any(|n| matches!(n, Node::Label(_))),
            "the label is gone: {items:?}"
        );
    }

    #[test]
    fn a_branch_to_the_next_statement_goes_away() {
        let mut items = body(vec![Node::Goto("L1".into()), Node::Label("L1".into())]);
        reshape(&mut items);
        assert!(items.is_empty(), "the branch and its label are gone");
    }

    #[test]
    fn a_branch_that_leaves_a_loop_becomes_a_break() {
        let mut items = body(vec![
            Node::DoWhile {
                body: vec![branch(compare(CondOp::Eq), "L1")],
                cond: Expr::Const(1),
            },
            Node::Label("L1".into()),
        ]);
        reshape(&mut items);
        let Node::DoWhile { body, .. } = &items[0] else {
            panic!("the loop is still a loop: {items:?}");
        };
        let Node::If { then_body, .. } = &body[0] else {
            panic!("the branch is still a branch: {body:?}");
        };
        assert!(
            matches!(then_body[0], Node::Break),
            "the branch is a break: {body:?}"
        );
    }

    #[test]
    fn a_label_another_branch_needs_is_kept() {
        let mut items = body(vec![
            branch(compare(CondOp::Sge), "L1"),
            Node::Assign {
                dest: "x".into(),
                value: Expr::Const(1),
            },
            Node::Label("L1".into()),
            Node::Goto("L1".into()),
        ]);
        reshape(&mut items);
        assert!(
            items.iter().any(|n| matches!(n, Node::Label(_))),
            "the label is still there: {items:?}"
        );
    }
}
