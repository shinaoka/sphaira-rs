//! Line-oriented keyword/block lexer shared by the PAO and VPS readers.
//!
//! Implements the lexical structure of `pao-basis.md` ("PAO file format", "Lexical structure") and
//! `nonlocal-pseudopotential.md` §2.1: first-word keywords (case-insensitive, first occurrence wins),
//! blocks `<name` ... `name>`, `#` comments.

use crate::error::AtomDataError;

/// A `<name` ... `name>` block; `lines` holds (1-based line number, comment-stripped text).
#[derive(Debug)]
pub(crate) struct Block<'a> {
    pub name: String,
    pub lines: Vec<(usize, &'a str)>,
}

/// Lexed document: top-level keywords and blocks.
#[derive(Debug)]
pub(crate) struct Document<'a> {
    keywords: Vec<(String, usize, &'a str)>,
    blocks: Vec<Block<'a>>,
}

fn strip_comment(s: &str) -> &str {
    s.split('#').next().unwrap_or("").trim()
}

impl<'a> Document<'a> {
    pub(crate) fn parse(text: &'a str) -> Self {
        let lines: Vec<(usize, &str)> = text
            .lines()
            .enumerate()
            .map(|(i, l)| (i + 1, strip_comment(l)))
            .collect();
        let mut keywords: Vec<(String, usize, &str)> = Vec::new();
        let mut blocks = Vec::new();
        let mut i = 0;
        while i < lines.len() {
            let (no, line) = lines[i];
            if let Some(tok) = line.split_whitespace().next() {
                if tok.len() > 1 && tok.starts_with('<') && !tok.ends_with('>') {
                    let name = &tok[1..];
                    let end = format!("{name}>");
                    let found = (i + 1..lines.len()).find(|&j| {
                        lines[j]
                            .1
                            .split_whitespace()
                            .next()
                            .is_some_and(|t| t.eq_ignore_ascii_case(&end))
                    });
                    if let Some(j) = found {
                        blocks.push(Block {
                            name: name.to_ascii_lowercase(),
                            lines: lines[i + 1..j]
                                .iter()
                                .copied()
                                .filter(|(_, l)| !l.is_empty())
                                .collect(),
                        });
                        i = j + 1;
                        continue;
                    }
                }
                let key = tok.to_ascii_lowercase();
                if !keywords.iter().any(|(k, _, _)| *k == key) {
                    keywords.push((key, no, line[tok.len()..].trim()));
                }
            }
            i += 1;
        }
        Self { keywords, blocks }
    }

    /// First occurrence of a keyword: (line number, rest of line).
    pub(crate) fn keyword(&self, name: &str) -> Option<(usize, &'a str)> {
        let key = name.to_ascii_lowercase();
        self.keywords
            .iter()
            .find(|(k, _, _)| *k == key)
            .map(|(_, n, v)| (*n, *v))
    }

    /// First block with this (case-insensitive) name.
    pub(crate) fn block(&self, name: &str) -> Option<&Block<'a>> {
        let key = name.to_ascii_lowercase();
        self.blocks.iter().find(|b| b.name == key)
    }

    pub(crate) fn blocks_with_prefix(&self, prefix: &str) -> impl Iterator<Item = &Block<'a>> {
        let p = prefix.to_ascii_lowercase();
        self.blocks.iter().filter(move |b| b.name.starts_with(&p))
    }

    pub(crate) fn required_block(
        &self,
        kind: &'static str,
        name: &str,
    ) -> Result<&Block<'a>, AtomDataError> {
        self.block(name).ok_or_else(|| AtomDataError::MissingBlock {
            file_kind: kind,
            block: name.to_string(),
        })
    }

    pub(crate) fn required_value<T: std::str::FromStr>(
        &self,
        kind: &'static str,
        name: &str,
        expected: &'static str,
    ) -> Result<T, AtomDataError> {
        self.optional_value(kind, name, expected)?
            .ok_or_else(|| AtomDataError::MissingKeyword {
                file_kind: kind,
                keyword: name.to_string(),
            })
    }

    pub(crate) fn optional_value<T: std::str::FromStr>(
        &self,
        kind: &'static str,
        name: &str,
        expected: &'static str,
    ) -> Result<Option<T>, AtomDataError> {
        let Some((line, rest)) = self.keyword(name) else {
            return Ok(None);
        };
        let tok = rest.split_whitespace().next().unwrap_or("");
        tok.parse::<T>()
            .map(Some)
            .map_err(|_| AtomDataError::Parse {
                file_kind: kind,
                line,
                text: tok.to_string(),
                expected,
            })
    }

    pub(crate) fn optional_bool(
        &self,
        kind: &'static str,
        name: &str,
    ) -> Result<Option<bool>, AtomDataError> {
        let Some((line, rest)) = self.keyword(name) else {
            return Ok(None);
        };
        let tok = rest.split_whitespace().next().unwrap_or("");
        match tok.to_ascii_lowercase().as_str() {
            "on" | "yes" | "true" | "t" | "y" | "1" => Ok(Some(true)),
            "off" | "no" | "false" | "f" | "n" | "0" => Ok(Some(false)),
            _ => Err(AtomDataError::Parse {
                file_kind: kind,
                line,
                text: tok.to_string(),
                expected: "a logical value (on/off, yes/no, true/false)",
            }),
        }
    }
}

/// Parses all tokens of a line as `f64`.
pub(crate) fn parse_floats(
    kind: &'static str,
    line: usize,
    text: &str,
) -> Result<Vec<f64>, AtomDataError> {
    text.split_whitespace()
        .map(|t| {
            t.parse::<f64>().map_err(|_| AtomDataError::Parse {
                file_kind: kind,
                line,
                text: t.to_string(),
                expected: "a real number",
            })
        })
        .collect()
}

/// Parses every line of a block as a row of exactly `cols` numbers.
pub(crate) fn parse_rows(
    kind: &'static str,
    block: &Block<'_>,
    cols: usize,
) -> Result<Vec<Vec<f64>>, AtomDataError> {
    block
        .lines
        .iter()
        .map(|&(no, l)| {
            let row = parse_floats(kind, no, l)?;
            if row.len() != cols {
                return Err(AtomDataError::Columns {
                    file_kind: kind,
                    line: no,
                    expected: cols,
                    found: row.len(),
                });
            }
            Ok(row)
        })
        .collect()
}

/// Splits a table (rows of `x, r, values...`) into the grid and value columns.
pub(crate) fn table_to_columns(rows: &[Vec<f64>]) -> (Vec<f64>, Vec<f64>, Vec<Vec<f64>>) {
    let xs = rows.iter().map(|r| r[0]).collect();
    let rs = rows.iter().map(|r| r[1]).collect();
    let ncol = rows.first().map_or(0, |r| r.len() - 2);
    let cols = (0..ncol)
        .map(|c| rows.iter().map(|r| r[2 + c]).collect())
        .collect();
    (xs, rs, cols)
}

#[cfg(test)]
mod tests;
