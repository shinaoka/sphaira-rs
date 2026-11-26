use super::*;

#[test]
fn keywords_first_wins_case_insensitive_and_comments() {
    let d = Document::parse("Foo.Bar 3 # c\nfoo.bar 4\n# only comment\n<VPS> L=0 x\nAbc on\n");
    assert_eq!(d.keyword("foo.BAR").unwrap().1, "3");
    assert_eq!(d.optional_bool("PAO", "abc").unwrap(), Some(true));
    assert!(d.keyword("<vps>").is_some());
}

#[test]
fn blocks_skip_inner_keywords_and_unterminated_is_plain_line() {
    let d = Document::parse("<Blk\n k 1\n 1 2\nblk>\nk 5\n<open\nz 1\n");
    assert_eq!(d.block("BLK").unwrap().lines.len(), 2);
    assert_eq!(d.keyword("k").unwrap().1, "5");
    assert!(d.block("open").is_none());
    assert!(d.keyword("<open").is_some());
}

#[test]
fn rows_check_columns() {
    let d = Document::parse("<t\n1 2\n3\nt>\n");
    let err = parse_rows("PAO", d.block("t").unwrap(), 2).unwrap_err();
    assert!(matches!(
        err,
        AtomDataError::Columns {
            line: 3,
            expected: 2,
            found: 1,
            ..
        }
    ));
}
