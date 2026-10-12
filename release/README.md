# The release guard

Results computed on PHercParis4 data are not published in this repository. `tests/test_no_paris4_figures.py`
checks that before each release:

1. `release/figures.py` reduces every number written in a text file to one canonical string (its rule is in
   the module docstring: absolute value, plain positional notation, percentages divided by 100, thousands
   separators dropped).
2. The test reduces every number in every tracked text file, in every file staged in the index, and in every
   file and message of every commit on any ref (`git rev-list --all`), and looks up the sha256 of each
   canonical string in a private list of the hashes of the withheld figures.
3. A match fails the test unless a private allow-list names that number on that line of that file.
4. The repository's object store must hold no unreachable object (`git fsck --unreachable --no-reflogs` prints
   nothing). A copy, archive or plain local clone of the directory carries every object in it, including old
   versions of files that were staged and replaced, which no commit holds and the scans above never read. Before
   a release, run `git reflog expire --expire=now --all && git gc --prune=now`, and publish by pushing (or from
   a `git clone --no-local`), never by copying the directory.

The guard checks numbers only. It does not replace a reading pass (below).

## The private files

Two files stay on the maintainers' machine and are never committed. `.gitignore` lists both, and a test fails if
either is tracked, staged or in any commit:

- `release/figure_hashes.json` (or the file named by `APOK_EVAL_FIGURE_HASHES`): the hashes. They are of short
  numeric strings, which can be inverted by trying every number, so the list is as private as the figures it
  stands for.
- `release/figure_allowlist.txt` (or the file named by `APOK_EVAL_FIGURE_ALLOWLIST`): the reviewed
  coincidences. Every entry is a number in the hash list, so publishing the allow-list would confirm each of
  its numbers as a withheld figure.

A clone of this repository has neither. There the scanning tests and the object-store check skip; the tests of
the normaliser, of the allow-list's format (on a synthetic example) and that neither file is tracked still run.
To run the guard, put both files in `release/` (or name them in the two variables) and run
`python -m pytest tests/test_no_paris4_figures.py`.

## The allow-list

A list of every figure of many runs covers much of the space of short numbers (most two- and three-digit
decimals between 0.1 and 1 are in it), so many numbers in code, tests and documentation match by coincidence.
Each one is reviewed and named where it occurs:

```
FORM | PATH | LINE | REASON
```

`FORM` is the canonical string, `PATH` the file's path in the repository, `LINE` the first 16 hex digits of the
sha256 of the line the number is on (stripped of leading and trailing whitespace), and `REASON` what the number
is there: a rule constant, a synthetic test value, a format constant, a line number in a citation, a year. An
entry allows that number on that line of that file and nowhere else. The same number on a new line, even in a
file where it is already allowed, fails, and so does any edit of an allowed line, until a maintainer has read
the occurrence and added its entry (a failure prints the line's `LINE` value). An entry that no file or commit
uses any more fails too, so the list stays exact. A number is allowed only when it is not a result computed on
PHercParis4 data.

## What the guard does not see

The hash list holds, from the sealed records of the withheld runs, the integers of 100 and over, the decimals with
at least two significant digits (and their roundings), and the runs' slab bounds and numeric identifiers. It leaves
out the configured constants (draw counts, thresholds, margins), which are the code's own defaults, and by its
rule it cannot hold the following, which the guard therefore does not catch:

- integers under 100, such as a count of fibres or of chunks;
- numbers with one significant digit after canonicalisation, such as a figure rounded to one digit;
- figures written in words, or as a rounded range in prose;
- dates, and the names of runs and data sets.

Each release therefore also gets a reading pass over everything that changed since the last one. A starting
point, each hit read in context:

```sh
git grep -n -E '\[ *[0-9]{3,6} *, *[0-9]{3,6} *[])]'            # slab and z ranges
git grep -n -i -E '\b[0-9]+ (fibres|fibers|chunks|segments|windings|points)\b'
git grep -n -E '\bR *[=≈~]? *0\.[0-9]'                         # locking strengths
git grep -n -i -E 'PHerc|Paris ?4|scroll ?1\b'                   # anything said about the scroll
```
