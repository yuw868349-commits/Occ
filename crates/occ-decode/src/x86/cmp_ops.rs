/// The comparison instructions, one row per family.
///
/// The name of each instruction is the condition it tests, so the
/// immediate byte selects the name. Every family was read out of a
/// disassembler; the script that does this is
/// `scripts/compare_tables.py`.
pub const CMP_OPS: &[CmpOp] = &[
    cmp_op(0xc2, 1, 0, false, ["vcmpeqps", "vcmpltps", "vcmpleps", "vcmpunordps", "vcmpneqps", "vcmpnltps", "vcmpnleps", "vcmpordps"]),
    cmp_op(0xc2, 1, 1, true, ["vcmpeqpd", "vcmpltpd", "vcmplepd", "vcmpunordpd", "vcmpneqpd", "vcmpnltpd", "vcmpnlepd", "vcmpordpd"]),
    cmp_op(0xc2, 1, 2, false, ["vcmpeqss", "vcmpltss", "vcmpless", "vcmpunordss", "vcmpneqss", "vcmpnltss", "vcmpnless", "vcmpordss"]),
    cmp_op(0xc2, 1, 3, true, ["vcmpeqsd", "vcmpltsd", "vcmplesd", "vcmpunordsd", "vcmpneqsd", "vcmpnltsd", "vcmpnlesd", "vcmpordsd"]),
    cmp_op(0x1e, 3, 1, false, ["vpcmpequd", "vpcmpltud", "vpcmpleud", "vpcmpud", "vpcmpnequd", "vpcmpnltud", "vpcmpnleud", "vpcmpud"]),
    cmp_op(0x1e, 3, 1, true, ["vpcmpequq", "vpcmpltuq", "vpcmpleuq", "vpcmpuq", "vpcmpnequq", "vpcmpnltuq", "vpcmpnleuq", "vpcmpuq"]),
    cmp_op(0x1f, 3, 1, false, ["vpcmpeqd", "vpcmpltd", "vpcmpled", "vpcmpd", "vpcmpneqd", "vpcmpnltd", "vpcmpnled", "vpcmpd"]),
    cmp_op(0x1f, 3, 1, true, ["vpcmpeqq", "vpcmpltq", "vpcmpleq", "vpcmpq", "vpcmpneqq", "vpcmpnltq", "vpcmpnleq", "vpcmpq"]),
    cmp_op(0x3e, 3, 1, false, ["vpcmpequb", "vpcmpltub", "vpcmpleub", "vpcmpub", "vpcmpnequb", "vpcmpnltub", "vpcmpnleub", "vpcmpub"]),
    cmp_op(0x3e, 3, 1, true, ["vpcmpequw", "vpcmpltuw", "vpcmpleuw", "vpcmpuw", "vpcmpnequw", "vpcmpnltuw", "vpcmpnleuw", "vpcmpuw"]),
    cmp_op(0x3f, 3, 1, false, ["vpcmpeqb", "vpcmpltb", "vpcmpleb", "vpcmpb", "vpcmpneqb", "vpcmpnltb", "vpcmpnleb", "vpcmpb"]),
    cmp_op(0x3f, 3, 1, true, ["vpcmpeqw", "vpcmpltw", "vpcmplew", "vpcmpw", "vpcmpneqw", "vpcmpnltw", "vpcmpnlew", "vpcmpw"]),
    cmp_op(0xc2, 3, 0, false, ["vcmpeqph", "vcmpltph", "vcmpleph", "vcmpunordph", "vcmpneqph", "vcmpnltph", "vcmpnleph", "vcmpordph"]),
    cmp_op(0xc2, 3, 2, false, ["vcmpeqsh", "vcmpltsh", "vcmplesh", "vcmpunordsh", "vcmpneqsh", "vcmpnltsh", "vcmpnlesh", "vcmpordsh"]),
    cmp_op(0xc2, 3, 3, false, ["vcmpeqbf16", "vcmpltbf16", "vcmplebf16", "vcmpunordbf16", "vcmpneqbf16", "vcmpnltbf16", "vcmpnlebf16", "vcmpordbf16"]),
];
