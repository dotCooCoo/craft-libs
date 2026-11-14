//! Platform regression guards checking source code for known defects and lifecycle hazards.

use std::collections::BTreeSet;
use std::fmt;
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

/// A diagnostic finding reported by a source guard.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finding {
    /// File path where the violation was detected.
    pub path: String,
    /// 1-based line number of the violation.
    pub line: usize,
    /// Diagnostic context (the offending code snippet or float literal).
    pub text: String,
}

impl Finding {
    /// Create a new diagnostic finding.
    pub fn new(path: impl Into<String>, line: usize, text: impl Into<String>) -> Self {
        Self { path: path.into(), line, text: text.into() }
    }
}

impl fmt::Display for Finding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.path.is_empty() { write!(f, "{}: {}", self.line, self.text) } else { write!(f, "{}:{}: {}", self.path, self.line, self.text) }
    }
}

/// Abstraction over a source file path and its text content.
pub trait AsFileSource {
    /// Path of the source file.
    fn path(&self) -> &str;
    /// Text content of the source file.
    fn content(&self) -> &str;
}

impl AsFileSource for (String, String) {
    fn path(&self) -> &str {
        &self.0
    }
    fn content(&self) -> &str {
        &self.1
    }
}

impl AsFileSource for &(String, String) {
    fn path(&self) -> &str {
        &self.0
    }
    fn content(&self) -> &str {
        &self.1
    }
}

impl AsFileSource for (&str, &str) {
    fn path(&self) -> &str {
        self.0
    }
    fn content(&self) -> &str {
        self.1
    }
}

impl AsFileSource for &(&str, &str) {
    fn path(&self) -> &str {
        self.0
    }
    fn content(&self) -> &str {
        self.1
    }
}

impl AsFileSource for (String, &str) {
    fn path(&self) -> &str {
        &self.0
    }
    fn content(&self) -> &str {
        self.1
    }
}

impl AsFileSource for &(String, &str) {
    fn path(&self) -> &str {
        &self.0
    }
    fn content(&self) -> &str {
        self.1
    }
}

impl AsFileSource for (&str, String) {
    fn path(&self) -> &str {
        self.0
    }
    fn content(&self) -> &str {
        &self.1
    }
}

impl AsFileSource for &(&str, String) {
    fn path(&self) -> &str {
        self.0
    }
    fn content(&self) -> &str {
        &self.1
    }
}

/// The deepest `use` group nesting and `cfg` predicate nesting the guards follow. Deeper input is
/// read as far as this depth, so hostile nesting cannot exhaust the stack.
const MAX_NESTING: usize = 64;

/// The kind of a Rust token. Comments are dropped; string, character and number literals keep no text
/// except string contents, which `cfg` predicates compare.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Kind {
    Ident(String),
    Punct(char),
    Str(String),
    Other,
}

#[derive(Debug, Clone)]
struct Token {
    kind: Kind,
    line: usize,
}

impl Token {
    fn is_ident(&self, name: &str) -> bool {
        matches!(&self.kind, Kind::Ident(s) if s == name)
    }

    fn is_punct(&self, c: char) -> bool {
        self.kind == Kind::Punct(c)
    }
}

fn is_ident_start(c: char) -> bool {
    c == '_' || c.is_alphabetic()
}

fn is_ident_continue(c: char) -> bool {
    c == '_' || c.is_alphanumeric()
}

/// Split Rust source into tokens with 1-based line numbers.
///
/// Line comments, nested block comments, string literals (normal, byte, C and raw strings with any
/// number of `#`), character and byte literals, lifetimes and raw identifiers are recognized, so a quote
/// or comment marker inside one of them does not hide the code after it. Unterminated literals and
/// comments run to the end of the input.
fn lex_rust(code: &str) -> Vec<Token> {
    let chars: Vec<char> = code.chars().collect();
    let len = chars.len();
    let at = |i: usize| chars.get(i).copied();
    let mut tokens = Vec::new();
    let mut i = 0;
    let mut line = 1usize;

    while i < len {
        let c = chars[i];
        if c == '\n' {
            line += 1;
            i += 1;
        } else if c.is_whitespace() {
            i += 1;
        } else if c == '/' && at(i + 1) == Some('/') {
            while i < len && chars[i] != '\n' {
                i += 1;
            }
        } else if c == '/' && at(i + 1) == Some('*') {
            let mut depth = 1usize;
            i += 2;
            while i < len && depth > 0 {
                if chars[i] == '/' && at(i + 1) == Some('*') {
                    depth += 1;
                    i += 2;
                } else if chars[i] == '*' && at(i + 1) == Some('/') {
                    depth -= 1;
                    i += 2;
                } else {
                    if chars[i] == '\n' {
                        line += 1;
                    }
                    i += 1;
                }
            }
        } else if is_ident_start(c) {
            let start = i;
            while i < len && is_ident_continue(chars[i]) {
                i += 1;
            }
            let word: String = chars[start..i].iter().collect();
            let token_line = line;
            if matches!(word.as_str(), "r" | "br" | "cr") && matches!(at(i), Some('"' | '#')) {
                let mut j = i;
                while j < len && chars[j] == '#' {
                    j += 1;
                }
                let hashes = j - i;
                if at(j) == Some('"') {
                    let (end, text) = scan_raw_string(&chars, j + 1, hashes, &mut line);
                    tokens.push(Token { kind: Kind::Str(text), line: token_line });
                    i = end;
                    continue;
                }
                if word == "r" && hashes == 1 && at(j).is_some_and(is_ident_start) {
                    let name_start = j;
                    while j < len && is_ident_continue(chars[j]) {
                        j += 1;
                    }
                    tokens.push(Token { kind: Kind::Ident(chars[name_start..j].iter().collect()), line: token_line });
                    i = j;
                    continue;
                }
            }
            if matches!(word.as_str(), "b" | "c") && at(i) == Some('"') {
                let (end, text) = scan_string(&chars, i + 1, &mut line);
                tokens.push(Token { kind: Kind::Str(text), line: token_line });
                i = end;
                continue;
            }
            if word == "b"
                && at(i) == Some('\'')
                && let Some(end) = char_literal_end(&chars, i)
            {
                tokens.push(Token { kind: Kind::Other, line: token_line });
                i = end;
                continue;
            }
            tokens.push(Token { kind: Kind::Ident(word), line: token_line });
        } else if c.is_ascii_digit() {
            while i < len && (is_ident_continue(chars[i]) || (chars[i] == '.' && at(i + 1).is_some_and(|d| d.is_ascii_digit()))) {
                i += 1;
            }
            tokens.push(Token { kind: Kind::Other, line });
        } else if c == '"' {
            let token_line = line;
            let (end, text) = scan_string(&chars, i + 1, &mut line);
            tokens.push(Token { kind: Kind::Str(text), line: token_line });
            i = end;
        } else if c == '\'' {
            if let Some(end) = char_literal_end(&chars, i) {
                i = end;
            } else {
                // A lifetime or a loop label.
                i += 1;
                while i < len && is_ident_continue(chars[i]) {
                    i += 1;
                }
            }
            tokens.push(Token { kind: Kind::Other, line });
        } else {
            tokens.push(Token { kind: Kind::Punct(c), line });
            i += 1;
        }
    }
    tokens
}

/// Read a normal string body starting after its opening quote. Returns the index after the closing
/// quote and the body as written, escapes included.
fn scan_string(chars: &[char], mut i: usize, line: &mut usize) -> (usize, String) {
    let start = i;
    while i < chars.len() {
        match chars[i] {
            '\\' => {
                if chars.get(i + 1) == Some(&'\n') {
                    *line += 1;
                }
                i += 2;
            }
            '"' => return (i + 1, chars[start..i].iter().collect()),
            '\n' => {
                *line += 1;
                i += 1;
            }
            _ => i += 1,
        }
    }
    (chars.len(), chars[start.min(chars.len())..].iter().collect())
}

/// Read a raw string body starting after its opening quote; it ends at a quote followed by `hashes` `#`.
fn scan_raw_string(chars: &[char], mut i: usize, hashes: usize, line: &mut usize) -> (usize, String) {
    let start = i;
    while i < chars.len() {
        if chars[i] == '"' && (1..=hashes).all(|h| chars.get(i + h) == Some(&'#')) {
            return (i + 1 + hashes, chars[start..i].iter().collect());
        }
        if chars[i] == '\n' {
            *line += 1;
        }
        i += 1;
    }
    (chars.len(), chars[start.min(chars.len())..].iter().collect())
}

/// The index after a character or byte literal whose opening quote is at `quote`, or `None` when the
/// quote starts a lifetime or a label.
fn char_literal_end(chars: &[char], quote: usize) -> Option<usize> {
    match chars.get(quote + 1) {
        Some('\\') => {
            // An escape: `\n`, `\'`, `\x7f`, `\u{10FFFF}`. The longest is twelve characters.
            let mut j = quote + 3;
            while j < chars.len() && j <= quote + 12 && chars[j] != '\'' && chars[j] != '\n' {
                j += 1;
            }
            (chars.get(j) == Some(&'\'')).then_some(j + 1)
        }
        Some('\n') | None => None,
        Some(_) => (chars.get(quote + 2) == Some(&'\'')).then_some(quote + 3),
    }
}

/// Rust keywords that can come right before a path that starts with `::`, as in `return ::std::time`.
/// `crate`, `self`, `super` and `Self` are path segments and are not listed.
const KEYWORDS: &[&str] = &[
    "as", "async", "await", "box", "break", "const", "continue", "dyn", "else", "enum", "extern", "fn", "for", "if", "impl", "in", "let", "loop",
    "match", "mod", "move", "mut", "pub", "ref", "return", "static", "struct", "trait", "type", "unsafe", "use", "where", "while", "yield",
];

/// Whether `toks[i]` follows `::` that itself follows a path segment, as `time` does in `a::time`.
fn continues_a_path(toks: &[Token], i: usize) -> bool {
    i >= 3
        && toks[i - 1].is_punct(':')
        && toks[i - 2].is_punct(':')
        && match &toks[i - 3].kind {
            Kind::Ident(s) => !KEYWORDS.contains(&s.as_str()),
            Kind::Punct('>') => true,
            _ => false,
        }
}

/// Whether `toks[i..]` starts with `::` followed by the identifier `name`.
fn then_path(toks: &[Token], i: usize, name: impl Fn(&str) -> bool) -> bool {
    toks.get(i).is_some_and(|t| t.is_punct(':'))
        && toks.get(i + 1).is_some_and(|t| t.is_punct(':'))
        && toks.get(i + 2).is_some_and(|t| matches!(&t.kind, Kind::Ident(s) if name(s)))
}

/// One name a `use` declaration brings into scope.
#[derive(Debug)]
struct UseLeaf {
    /// The full path, `self` removed and without a leading `::`.
    path: Vec<String>,
    /// The name it is bound to; `None` for a glob import.
    name: Option<String>,
    line: usize,
    /// Index of the `use` keyword.
    start: usize,
}

/// Every name the `use` declarations of a file bring into scope.
fn use_leaves(toks: &[Token]) -> Vec<UseLeaf> {
    let mut leaves = Vec::new();
    let mut i = 0;
    while i < toks.len() {
        if toks[i].is_ident("use") && !continues_a_path(toks, i) {
            let start = i;
            let end = (i + 1..toks.len()).find(|&j| toks[j].is_punct(';')).unwrap_or(toks.len());
            let mut pos = i + 1;
            let mut prefix = Vec::new();
            parse_use_tree(&toks[..end], &mut pos, &mut prefix, start, &mut leaves, 0);
            i = end;
        } else {
            i += 1;
        }
    }
    leaves
}

fn parse_use_tree(toks: &[Token], pos: &mut usize, prefix: &mut Vec<String>, start: usize, out: &mut Vec<UseLeaf>, depth: usize) {
    let base = prefix.len();
    if toks.get(*pos).is_some_and(|t| t.is_punct(':')) && toks.get(*pos + 1).is_some_and(|t| t.is_punct(':')) {
        *pos += 2;
    }
    while let Some(tok) = toks.get(*pos) {
        match &tok.kind {
            Kind::Ident(name) => {
                *pos += 1;
                if toks.get(*pos).is_some_and(|t| t.is_punct(':')) && toks.get(*pos + 1).is_some_and(|t| t.is_punct(':')) {
                    prefix.push(name.clone());
                    *pos += 2;
                    continue;
                }
                let mut path = prefix.clone();
                if name != "self" {
                    path.push(name.clone());
                }
                let mut bound = path.last().cloned();
                if toks.get(*pos).is_some_and(|t| t.is_ident("as"))
                    && let Some(Kind::Ident(alias)) = toks.get(*pos + 1).map(|t| &t.kind)
                {
                    bound = Some(alias.clone());
                    *pos += 2;
                }
                if let Some(bound) = bound.filter(|b| b != "_") {
                    out.push(UseLeaf { path, name: Some(bound), line: tok.line, start });
                }
                break;
            }
            Kind::Punct('*') => {
                out.push(UseLeaf { path: prefix.clone(), name: None, line: tok.line, start });
                *pos += 1;
                break;
            }
            Kind::Punct('{') => {
                *pos += 1;
                while let Some(t) = toks.get(*pos) {
                    if t.is_punct('}') {
                        *pos += 1;
                        break;
                    }
                    let before = *pos;
                    if !t.is_punct(',') && depth < MAX_NESTING {
                        parse_use_tree(toks, pos, prefix, start, out, depth + 1);
                    }
                    if *pos == before {
                        *pos += 1;
                    }
                }
                break;
            }
            _ => break,
        }
    }
    prefix.truncate(base);
}

/// A value of a `cfg` predicate on `wasm32-unknown-unknown`, where the value of a feature or of
/// `test` is not known from the source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tri {
    True,
    False,
    Unknown,
}

impl Tri {
    fn from_bool(b: bool) -> Self {
        if b { Self::True } else { Self::False }
    }

    fn not(self) -> Self {
        match self {
            Self::True => Self::False,
            Self::False => Self::True,
            Self::Unknown => Self::Unknown,
        }
    }
}

/// Evaluate the `cfg` predicate at `toks[*pos]` for `wasm32-unknown-unknown`.
fn eval_cfg(toks: &[Token], pos: &mut usize, depth: usize) -> Tri {
    let Some(Kind::Ident(name)) = toks.get(*pos).map(|t| &t.kind) else {
        return Tri::Unknown;
    };
    *pos += 1;
    if toks.get(*pos).is_some_and(|t| t.is_punct('=')) {
        let value = match toks.get(*pos + 1).map(|t| &t.kind) {
            Some(Kind::Str(v)) => v.clone(),
            _ => return Tri::Unknown,
        };
        *pos += 2;
        let expected = match name.as_str() {
            "target_arch" => "wasm32",
            "target_os" | "target_vendor" => "unknown",
            "target_family" => "wasm",
            "target_env" => "",
            "target_pointer_width" => "32",
            "target_endian" => "little",
            _ => return Tri::Unknown,
        };
        return Tri::from_bool(value == expected);
    }
    if toks.get(*pos).is_some_and(|t| t.is_punct('(')) {
        *pos += 1;
        let mut values = Vec::new();
        while let Some(t) = toks.get(*pos) {
            if t.is_punct(')') {
                *pos += 1;
                break;
            }
            if t.is_punct(',') || depth >= MAX_NESTING {
                *pos += 1;
                continue;
            }
            let before = *pos;
            values.push(eval_cfg(toks, pos, depth + 1));
            if *pos == before {
                *pos += 1;
            }
        }
        return match name.as_str() {
            "not" => values.first().map_or(Tri::Unknown, |v| v.not()),
            "all" if values.contains(&Tri::False) => Tri::False,
            "all" if values.contains(&Tri::Unknown) => Tri::Unknown,
            "all" => Tri::True,
            "any" if values.contains(&Tri::True) => Tri::True,
            "any" if values.contains(&Tri::Unknown) => Tri::Unknown,
            "any" => Tri::False,
            _ => Tri::Unknown,
        };
    }
    match name.as_str() {
        "unix" | "windows" => Tri::False,
        _ => Tri::Unknown,
    }
}

/// The index of the token that closes the bracket opened at `open`, or `toks.len()`.
fn matching_close(toks: &[Token], open: usize) -> usize {
    let mut depth = 0usize;
    for (j, t) in toks.iter().enumerate().skip(open) {
        match t.kind {
            Kind::Punct('(' | '[' | '{') => depth += 1,
            Kind::Punct(')' | ']' | '}') => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return j;
                }
            }
            _ => {}
        }
    }
    toks.len()
}

/// Keywords whose item or expression ends with its closing brace rather than with `;`.
const BLOCK_KEYWORDS: &[&str] =
    &["fn", "mod", "impl", "trait", "struct", "enum", "union", "extern", "macro_rules", "if", "match", "loop", "while", "for", "unsafe"];

/// The index after the item, statement, field or match arm that starts at `start`, or with `inner`, the
/// index of the bracket that closes the enclosing scope.
///
/// It ends after a `;` or a `,` outside brackets, after the closing brace of a block item (one that
/// starts with a keyword in [`BLOCK_KEYWORDS`] or is a macro call with braces) or of a match arm's
/// block, or before a closing bracket of the enclosing scope. A `,` inside the angle brackets of a
/// field's or a `let`'s type ends the item early, which leaves the rest of it checked.
fn item_end(toks: &[Token], start: usize, inner: bool) -> usize {
    let mut depth = 0usize;
    let mut block_item = false;
    let mut arm = false;
    let mut j = start;
    while j < toks.len() {
        let at_top = depth == 0 && !inner;
        match &toks[j].kind {
            Kind::Ident(k) if at_top && BLOCK_KEYWORDS.contains(&k.as_str()) => block_item = true,
            Kind::Punct('!') if at_top && toks.get(j + 1).is_some_and(|t| t.is_punct('{')) => block_item = true,
            Kind::Punct('=') if at_top && toks.get(j + 1).is_some_and(|t| t.is_punct('>')) => arm = true,
            Kind::Punct('(' | '[' | '{') => depth += 1,
            Kind::Punct(c @ (')' | ']' | '}')) => {
                if depth == 0 {
                    return j;
                }
                depth -= 1;
                if depth == 0 && *c == '}' && (block_item || arm) && !inner {
                    if toks.get(j + 1).is_some_and(|t| t.is_ident("else")) {
                        j += 1;
                        continue;
                    }
                    let comma = toks.get(j + 1).is_some_and(|t| t.is_punct(','));
                    return if arm && comma { j + 2 } else { j + 1 };
                }
            }
            Kind::Punct(';') if at_top => return j + 1,
            Kind::Punct(',') if at_top && (arm || !block_item) => return j + 1,
            _ => {}
        }
        j += 1;
    }
    toks.len()
}

/// Mark the tokens of every item whose `cfg` attribute is false on `wasm32-unknown-unknown`.
///
/// An outer attribute (`#[cfg(...)]`) covers the item after it; an inner attribute (`#![cfg(...)]`)
/// covers the rest of the module or block it is in.
fn native_only(toks: &[Token]) -> Vec<bool> {
    let mut skip = vec![false; toks.len()];
    let mut i = 0;
    while i < toks.len() {
        if !toks[i].is_punct('#') {
            i += 1;
            continue;
        }
        let inner = toks.get(i + 1).is_some_and(|t| t.is_punct('!'));
        let open = if inner { i + 2 } else { i + 1 };
        if !toks.get(open).is_some_and(|t| t.is_punct('[')) {
            i += 1;
            continue;
        }
        let close = matching_close(toks, open);
        if close >= toks.len() {
            // An unclosed attribute: no later bracket closes it, so nothing after it is an item.
            break;
        }
        let excluded = toks.get(open + 1).is_some_and(|t| t.is_ident("cfg")) && toks.get(open + 2).is_some_and(|t| t.is_punct('(')) && {
            let mut pos = open + 3;
            eval_cfg(&toks[..close], &mut pos, 0) == Tri::False
        };
        if !excluded {
            i = close + 1;
            continue;
        }
        // Further outer attributes on the same item.
        let mut item = close + 1;
        while !inner && item < toks.len() && toks[item].is_punct('#') && toks.get(item + 1).is_some_and(|t| t.is_punct('[')) {
            item = matching_close(toks, item + 1).saturating_add(1);
        }
        let end = item_end(toks, item.min(toks.len()), inner);
        for s in skip.iter_mut().take(end).skip(i) {
            *s = true;
        }
        i = end.max(i + 1);
    }
    skip
}

/// Add one finding for each line in `lines`, in line order, with the trimmed text of that line.
fn report(findings: &mut Vec<Finding>, path: &str, content: &str, lines: &BTreeSet<usize>) {
    if lines.is_empty() {
        return;
    }
    let text: Vec<&str> = content.lines().collect();
    for &line in lines {
        let line_text = text.get(line.saturating_sub(1)).map_or("", |s| s.trim());
        findings.push(Finding::new(path, line, line_text));
    }
}

/// Guard: detects direct calls to `muda`'s predefined Quit menu item.
///
/// Using `PredefinedMenuItem::quit` connects the macOS menu directly to `NSApplication terminate:`,
/// which bypasses dirty document checks, unsaved changes save prompts, and lifecycle cleanup.
/// Applications should route Quit through custom menu items or interceptors connected to `craft-quit`.
///
/// The guard reads Rust tokens, so comments, strings and character literals are ignored, and whitespace
/// or line breaks around `::` do not hide a call. It also follows a `use` that renames muda's type
/// (`use muda::PredefinedMenuItem as Item;` then `Item::quit`); a rename of another crate's
/// `PredefinedMenuItem` is not followed. Each finding names the line where the
/// type's name starts, once per line, in line order.
pub fn predefined_quit<I>(files: I) -> Vec<Finding>
where
    I: IntoIterator,
    I::Item: AsFileSource,
{
    let mut findings = Vec::new();
    for file in files {
        let path = file.path();
        let content = file.content();
        let toks = lex_rust(content);
        let aliases: BTreeSet<String> = use_leaves(&toks)
            .into_iter()
            .filter(|leaf| leaf.path == ["muda", "PredefinedMenuItem"])
            .filter_map(|leaf| leaf.name)
            .filter(|name| name != "PredefinedMenuItem")
            .collect();
        let mut lines = BTreeSet::new();
        for (i, tok) in toks.iter().enumerate() {
            let Kind::Ident(name) = &tok.kind else { continue };
            let named = name == "PredefinedMenuItem" || (aliases.contains(name) && !continues_a_path(&toks, i));
            if named && then_path(&toks, i + 1, |s| s == "quit" || s == "Quit") {
                lines.insert(tok.line);
            }
        }
        report(&mut findings, path, content, &lines);
    }
    findings
}

/// Guard: detects uses of `std::time::Instant` in code paths running on the web.
///
/// On `wasm32-unknown-unknown`, `std::time::Instant::now()` panics with "time not implemented
/// on this platform". Crates running in the browser should use `web_time::Instant`, which maps to
/// `std::time::Instant` on native platforms and `performance.now()` on the web.
///
/// ### Syntactic Scope and Limitations
///
/// * **Targeted files**: Callers should pass files from crates that compile into the application's web build.
/// * **What is reported**: the paths `std::time::Instant` and `::std::time::Instant`; `<m>::Instant` where a
///   `use` outside a native-only item binds `m` to `std::time` (`use std::time;`, `use std::time as t;`,
///   `use std::time::{self as t}`);
///   and every `use` that imports `std::time::Instant` or `std::time::*`, under any name. Each finding names
///   the line where the path starts, once per line, in line order.
/// * **What is skipped**: comments, string and character literals, other crates' `Instant` (`web_time::Instant`,
///   a `time` module that is not `std::time`), and items whose `cfg` attribute is false on `wasm32-unknown-unknown`
///   (for example `#[cfg(not(target_arch = "wasm32"))]`, `#[cfg(all(unix, ...))]`, `#[cfg(target_os = "macos")]`,
///   or `#![cfg(...)]` for the rest of a module). A predicate that depends on a feature or on `test` is not
///   known from the source, so the item is checked.
/// * **Limitations**: This is a lexical source guard. It does not expand macros, resolve paths through other
///   modules or re-exports, rename `std` itself (`use std as s;`), or follow aliases in a different scope than
///   the file.
pub fn std_instant<I>(files: I) -> Vec<Finding>
where
    I: IntoIterator,
    I::Item: AsFileSource,
{
    let mut findings = Vec::new();
    for file in files {
        let path = file.path();
        let content = file.content();
        let toks = lex_rust(content);
        let skip = native_only(&toks);
        let leaves = use_leaves(&toks);
        let modules: BTreeSet<String> = leaves
            .iter()
            .filter(|leaf| leaf.path == ["std", "time"] && !skip.get(leaf.start).copied().unwrap_or(false))
            .filter_map(|leaf| leaf.name.clone())
            .collect();
        let is_std_time = |path: &[String]| path == ["std", "time"] || (path.len() == 1 && modules.contains(&path[0]));

        let mut lines = BTreeSet::new();
        for leaf in &leaves {
            if skip.get(leaf.start).copied().unwrap_or(false) {
                continue;
            }
            let imports_instant = if leaf.name.is_none() {
                is_std_time(&leaf.path)
            } else {
                leaf.path.split_last().is_some_and(|(last, parent)| last == "Instant" && is_std_time(parent))
            };
            if imports_instant {
                lines.insert(leaf.line);
            }
        }
        for (i, tok) in toks.iter().enumerate() {
            if skip[i] || continues_a_path(&toks, i) {
                continue;
            }
            let Kind::Ident(name) = &tok.kind else { continue };
            let hit = if name == "std" {
                then_path(&toks, i + 1, |s| s == "time") && then_path(&toks, i + 4, |s| s == "Instant")
            } else {
                modules.contains(name) && then_path(&toks, i + 1, |s| s == "Instant")
            };
            if hit {
                lines.insert(tok.line);
            }
        }
        report(&mut findings, path, content, &lines);
    }
    findings
}

/// Replace WGSL comments (`//` and nested `/* ... */`) with spaces, keeping line breaks.
fn mask_wgsl_comments(code: &str) -> String {
    let chars: Vec<char> = code.chars().collect();
    let len = chars.len();
    let mut out = String::with_capacity(code.len());
    let mut i = 0;
    let mut depth = 0usize;
    let mut in_line = false;
    while i < len {
        let c = chars[i];
        let next = chars.get(i + 1).copied();
        if c == '\n' {
            in_line = false;
            out.push('\n');
            i += 1;
        } else if in_line {
            out.push(' ');
            i += 1;
        } else if c == '/' && next == Some('*') {
            depth += 1;
            out.push_str("  ");
            i += 2;
        } else if depth > 0 && c == '*' && next == Some('/') {
            depth -= 1;
            out.push_str("  ");
            i += 2;
        } else if depth > 0 {
            out.push(' ');
            i += 1;
        } else if c == '/' && next == Some('/') {
            in_line = true;
            out.push_str("  ");
            i += 2;
        } else {
            out.push(c);
            i += 1;
        }
    }
    out
}

/// The largest finite value of the type a WGSL float literal with this suffix has.
fn float_limit(suffix: Option<u8>) -> f64 {
    if suffix == Some(b'h') { 65504.0 } else { f64::from(f32::MAX) }
}

/// Read a WGSL hexadecimal literal starting at the `0` of `0x`. Returns the index after it, the length of
/// its text without a suffix, and whether it is a float above the range of its type.
fn scan_wgsl_hex(code: &[u8], start: usize) -> (usize, usize, bool) {
    // Significant hex digits kept; 15 hold 60 bits, more than an f64 mantissa.
    const KEPT: usize = 15;
    let mut i = start + 2;
    let mut mantissa = 0f64;
    let mut kept = 0usize;
    let mut exponent: i64 = 0;
    let mut digits = 0usize;
    let mut digit = |d: u8, fraction: bool, exponent: &mut i64| {
        let value = (d as char).to_digit(16).unwrap_or(0);
        if mantissa == 0.0 && value == 0 {
            if fraction {
                *exponent -= 4;
            }
        } else if kept < KEPT {
            mantissa = mantissa * 16.0 + f64::from(value);
            kept += 1;
            if fraction {
                *exponent -= 4;
            }
        } else if !fraction {
            *exponent += 4;
        }
    };
    while i < code.len() && code[i].is_ascii_hexdigit() {
        digit(code[i], false, &mut exponent);
        digits += 1;
        i += 1;
    }
    let mut is_float = false;
    if code.get(i) == Some(&b'.') {
        is_float = true;
        i += 1;
        while i < code.len() && code[i].is_ascii_hexdigit() {
            digit(code[i], true, &mut exponent);
            digits += 1;
            i += 1;
        }
    }
    let mut has_exponent = false;
    if matches!(code.get(i), Some(b'p' | b'P')) {
        let mut j = i + 1;
        let negative = code.get(j) == Some(&b'-');
        if matches!(code.get(j), Some(b'+' | b'-')) {
            j += 1;
        }
        if code.get(j).is_some_and(u8::is_ascii_digit) {
            let mut e: i64 = 0;
            while j < code.len() && code[j].is_ascii_digit() {
                e = (e * 10 + i64::from(code[j] - b'0')).min(1_000_000);
                j += 1;
            }
            exponent += if negative { -e } else { e };
            has_exponent = true;
            is_float = true;
            i = j;
        }
    }
    let text_len = i - start;
    let suffix = if has_exponent && matches!(code.get(i), Some(b'f' | b'h')) { code.get(i).copied() } else { None };
    let end = if suffix.is_some() || (!is_float && matches!(code.get(i), Some(b'i' | b'u'))) { i + 1 } else { i };
    if !is_float || digits == 0 || mantissa == 0.0 {
        return (end, text_len, false);
    }
    let scale = i32::try_from(exponent.clamp(-4000, 4000)).unwrap_or(0);
    let value = mantissa * 2f64.powi(scale);
    (end, text_len, !value.is_finite() || value > float_limit(suffix))
}

/// Guard: scans a single WGSL shader string and detects float literals outside their type's finite range.
///
/// WebGPU compilers in browsers reject a float literal above `f32::MAX` (e.g. `3.40282347e38`) with a
/// pipeline build error, which turns off GPU rendering. Exact values equal to `f32::MAX` can be written
/// as `3.4028234663852886e38` or as the hex float `0x1.fffffep+127f`.
///
/// ### Policy and Limitations
///
/// * **Conservative mathematical bound**: The guard compares the literal's value, rounded to `f64`,
///   with `f64::from(f32::MAX)`, or with 65504 (the largest finite `f16`) for an `h` suffix. Any literal
///   above it is reported, even one that a compiler would round down to the maximum.
/// * **Which literals**: decimal floats (with a `.`, an exponent or an `f`/`h` suffix) and hexadecimal floats
///   (`0x` with a `.` or a `p` exponent). Integer literals (`123`, `123u`, `0xFFi`, `0x1f`) are not floats and
///   are not reported. A `-` in front is not part of the literal, so `-1e39` is reported as `1e39`.
/// * **Comments**: Comments (`//` and nested `/* ... */`) are masked out prior to scanning.
pub fn wgsl_float_literals(wgsl: &str) -> Vec<Finding> {
    wgsl_float_literals_in([("", wgsl)])
}

/// Guard: scans multiple WGSL shader files and detects float literals outside their type's finite range,
/// with the rules of [`wgsl_float_literals`].
pub fn wgsl_float_literals_in<I>(files: I) -> Vec<Finding>
where
    I: IntoIterator,
    I::Item: AsFileSource,
{
    let mut findings = Vec::new();
    for file in files {
        let path = file.path();
        let masked = mask_wgsl_comments(file.content());

        for (n, line) in masked.lines().enumerate() {
            let code = line.as_bytes();
            let mut i = 0;
            while i < code.len() {
                let digit_or_dot = code[i].is_ascii_digit() || (code[i] == b'.' && code.get(i + 1).is_some_and(u8::is_ascii_digit));
                let word_before = i > 0 && (code[i - 1].is_ascii_alphanumeric() || code[i - 1] == b'_' || code[i - 1] == b'.');
                if !digit_or_dot || word_before {
                    i += 1;
                    continue;
                }
                let start = i;
                if code[i] == b'0' && matches!(code.get(i + 1), Some(b'x' | b'X')) {
                    let (end, text_len, over) = scan_wgsl_hex(code, i);
                    if over {
                        findings.push(Finding::new(path, n + 1, String::from_utf8_lossy(&code[start..start + text_len]).into_owned()));
                    }
                    i = end.max(i + 1);
                    continue;
                }
                let mut is_float = false;
                while i < code.len() && code[i].is_ascii_digit() {
                    i += 1;
                }
                if code.get(i) == Some(&b'.') {
                    is_float = true;
                    i += 1;
                    while i < code.len() && code[i].is_ascii_digit() {
                        i += 1;
                    }
                }
                if matches!(code.get(i), Some(b'e' | b'E')) {
                    let mut j = i + 1;
                    if matches!(code.get(j), Some(b'+' | b'-')) {
                        j += 1;
                    }
                    if code.get(j).is_some_and(u8::is_ascii_digit) {
                        is_float = true;
                        i = j;
                        while i < code.len() && code[i].is_ascii_digit() {
                            i += 1;
                        }
                    }
                }
                let literal = String::from_utf8_lossy(&code[start..i]).into_owned();
                let suffix = match code.get(i) {
                    Some(&s @ (b'f' | b'h')) => {
                        is_float = true;
                        i += 1;
                        Some(s)
                    }
                    Some(b'i' | b'u') if !is_float => {
                        i += 1;
                        None
                    }
                    _ => None,
                };
                if !is_float {
                    continue;
                }
                let is_overflow = match literal.parse::<f64>() {
                    Ok(v) => !v.is_finite() || v.abs() > float_limit(suffix),
                    Err(_) => true,
                };
                if is_overflow {
                    findings.push(Finding::new(path, n + 1, literal));
                }
            }
        }
    }
    findings
}

/// Size limits for [`rust_sources_with_limits`]. An input over a limit is an error.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceLimits {
    /// The largest single file read, in bytes. Default 8 MiB.
    pub max_file_bytes: u64,
    /// The largest total of all files read, in bytes. Default 256 MiB.
    pub max_total_bytes: u64,
    /// The most files read. Default 50,000.
    pub max_files: usize,
}

#[cfg(not(target_arch = "wasm32"))]
impl Default for SourceLimits {
    fn default() -> Self {
        Self { max_file_bytes: 8 << 20, max_total_bytes: 256 << 20, max_files: 50_000 }
    }
}

/// The first line of a `CACHEDIR.TAG` file, which marks a folder of generated files
/// (<https://bford.info/cachedir/>). Cargo writes one in each target folder.
#[cfg(not(target_arch = "wasm32"))]
const CACHEDIR_SIGNATURE: &[u8] = b"Signature: 8a477f597d28d172789f06886806bc55";

#[cfg(not(target_arch = "wasm32"))]
fn is_cache_dir(dir: &Path) -> bool {
    use std::io::Read;
    let Ok(file) = std::fs::File::open(dir.join("CACHEDIR.TAG")) else {
        return false;
    };
    let mut head = Vec::with_capacity(CACHEDIR_SIGNATURE.len());
    file.take(CACHEDIR_SIGNATURE.len() as u64).read_to_end(&mut head).is_ok() && head == CACHEDIR_SIGNATURE
}

/// Recursively collect all Rust source files under `dir`, excluding test files and test directories,
/// with the default [`SourceLimits`].
///
/// Excluded files:
/// * `tests.rs`
/// * `tests_*.rs`
/// * `*_tests.rs`
/// * `*_test.rs`
/// * `test.rs`
/// * Any file below `dir` within a directory named `tests`, `benches`, `examples` or `target`, or whose
///   name starts with `.`, or that holds a `CACHEDIR.TAG` file (a build output folder with another name).
/// * Symbolic links (and, on Windows, junctions) to files or folders, which are not followed.
///
/// Results are sorted lexicographically by file path for deterministic behavior across platforms.
#[cfg(not(target_arch = "wasm32"))]
pub fn rust_sources(dir: impl AsRef<Path>) -> std::io::Result<Vec<(String, String)>> {
    rust_sources_with_limits(dir, SourceLimits::default())
}

/// [`rust_sources`] with explicit size limits. A file over `max_file_bytes` returns an error of kind
/// [`std::io::ErrorKind::FileTooLarge`]; more than `max_files` files or `max_total_bytes` bytes in all
/// returns an error of kind [`std::io::ErrorKind::QuotaExceeded`].
#[cfg(not(target_arch = "wasm32"))]
pub fn rust_sources_with_limits(dir: impl AsRef<Path>, limits: SourceLimits) -> std::io::Result<Vec<(String, String)>> {
    use std::io::{Error, ErrorKind, Read};

    let root = dir.as_ref();
    let mut files = Vec::new();
    let mut total: u64 = 0;
    let mut dirs = vec![root.to_path_buf()];

    while let Some(d) = dirs.pop() {
        for entry in std::fs::read_dir(&d)? {
            let entry = entry?;
            let p = entry.path();
            let kind = entry.file_type()?;
            if kind.is_symlink() {
                continue;
            }
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if kind.is_dir() {
                if name.starts_with('.') || name == "target" || name == "tests" || name == "benches" || name == "examples" || is_cache_dir(&p) {
                    continue;
                }
                dirs.push(p);
            } else if kind.is_file() {
                if !name.ends_with(".rs") {
                    continue;
                }
                if name == "tests.rs" || name.starts_with("tests_") || name.ends_with("_tests.rs") || name.ends_with("_test.rs") || name == "test.rs"
                {
                    continue;
                }
                if files.len() >= limits.max_files {
                    return Err(Error::new(
                        ErrorKind::QuotaExceeded,
                        format!("more than {} Rust source files under {}", limits.max_files, root.display()),
                    ));
                }
                let over_file = || Error::new(ErrorKind::FileTooLarge, format!("{} is larger than {} bytes", p.display(), limits.max_file_bytes));
                if entry.metadata()?.len() > limits.max_file_bytes {
                    return Err(over_file());
                }
                let mut content = String::new();
                std::fs::File::open(&p)?.take(limits.max_file_bytes.saturating_add(1)).read_to_string(&mut content)?;
                let size = content.len() as u64;
                if size > limits.max_file_bytes {
                    return Err(over_file());
                }
                total = total.saturating_add(size);
                if total > limits.max_total_bytes {
                    return Err(Error::new(
                        ErrorKind::QuotaExceeded,
                        format!("Rust sources under {} are larger than {} bytes in all", root.display(), limits.max_total_bytes),
                    ));
                }
                files.push((p.display().to_string(), content));
            }
        }
    }

    files.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(files)
}
