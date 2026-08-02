use super::*;

// ─── Are the PYTHON oracles competent? ────────────────────────────────────────
//
// `checker_adequacy` did this for the Rust tasks and found two of twelve unable to
// reject a plausible wrong answer. The Python checkers had never been audited, and
// they are older — so this is the same mutation test applied to them.
//
// The check runs in both directions, because a checker fails in two ways:
//
//   1. It must ACCEPT a correct implementation. A checker that rejects valid code
//      turns a capable model into an apparent failure.
//   2. It must REJECT every plausible-but-wrong one. A checker that waves a mutant
//      through has silently approved a bug, and every score it ever produced was
//      partly noise.
//
// The mutants are deliberately the mistakes a model actually makes — off-by-one at a
// boundary, the wrong string-splitting call, one level of recursion instead of all —
// not absurdities. An oracle that only catches nonsense is not an oracle.

pub(crate) struct PyAdequacy {
    /// Must match the `name` of the task in `code_gen_bench`.
    pub(crate) task: &'static str,
    /// A correct implementation the checker must accept.
    pub(crate) reference: &'static str,
    /// (label, wrong implementation) — each must be rejected.
    pub(crate) mutants: &'static [(&'static str, &'static str)],
}

pub(crate) const PY_ADEQUACY: &[PyAdequacy] = &[
    PyAdequacy {
        task: "has_close_elements",
        reference: "def has_close_elements(numbers, threshold):\n    for i in range(len(numbers)):\n        for j in range(i + 1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n",
        mutants: &[
            // The classic strict/non-strict slip: the spec says STRICTLY less than.
            ("uses <= instead of <", "def has_close_elements(numbers, threshold):\n    for i in range(len(numbers)):\n        for j in range(i + 1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) <= threshold:\n                return True\n    return False\n"),
            // Only compares neighbours, so a close pair further apart is missed.
            ("only compares adjacent elements", "def has_close_elements(numbers, threshold):\n    for i in range(len(numbers) - 1):\n        if abs(numbers[i] - numbers[i + 1]) < threshold:\n            return True\n    return False\n"),
        ],
    },
    PyAdequacy {
        task: "sum_even_numbers",
        reference: "def sum_even_numbers(nums):\n    return sum(n for n in nums if n % 2 == 0)\n",
        mutants: &[
            ("sums the odd numbers", "def sum_even_numbers(nums):\n    return sum(n for n in nums if n % 2 == 1)\n"),
            // n % 2 is truthy for negative odds too, but the trap is that it sums
            // everything when the filter is dropped.
            ("sums everything", "def sum_even_numbers(nums):\n    return sum(nums)\n"),
        ],
    },
    PyAdequacy {
        task: "reverse_words",
        reference: "def reverse_words(s):\n    return ' '.join(reversed(s.split()))\n",
        mutants: &[
            // split(' ') differs from split() exactly on runs of whitespace — the
            // single most common way to get this almost right.
            ("splits on a single space, not whitespace", "def reverse_words(s):\n    return ' '.join(reversed(s.split(' ')))\n"),
            ("reverses the characters instead", "def reverse_words(s):\n    return s[::-1]\n"),
        ],
    },
    PyAdequacy {
        task: "is_prime",
        reference: "def is_prime(n):\n    if n < 2:\n        return False\n    i = 2\n    while i * i <= n:\n        if n % i == 0:\n            return False\n        i += 1\n    return True\n",
        mutants: &[
            ("calls 1 prime", "def is_prime(n):\n    if n < 1:\n        return False\n    i = 2\n    while i * i <= n:\n        if n % i == 0:\n            return False\n        i += 1\n    return True\n"),
            ("negative numbers come out prime", "def is_prime(n):\n    if n in (0, 1):\n        return False\n    i = 2\n    while i * i <= n:\n        if n % i == 0:\n            return False\n        i += 1\n    return True\n"),
        ],
    },
    PyAdequacy {
        task: "roman_to_int",
        reference: "def roman_to_int(s):\n    v = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}\n    total = 0\n    for i, c in enumerate(s):\n        if i + 1 < len(s) and v[c] < v[s[i+1]]:\n            total -= v[c]\n        else:\n            total += v[c]\n    return total\n",
        mutants: &[
            ("ignores the subtractive rule", "def roman_to_int(s):\n    v = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}\n    return sum(v[c] for c in s)\n"),
            ("only subtracts for I, not X or C", "def roman_to_int(s):\n    v = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}\n    total = 0\n    for i, c in enumerate(s):\n        if c == 'I' and i + 1 < len(s) and v[c] < v[s[i+1]]:\n            total -= v[c]\n        else:\n            total += v[c]\n    return total\n"),
        ],
    },
    PyAdequacy {
        task: "is_balanced_brackets",
        reference: "def is_balanced(s):\n    pairs = {')':'(', ']':'[', '}':'{'}\n    stack = []\n    for c in s:\n        if c in '([{':\n            stack.append(c)\n        elif c in pairs:\n            if not stack or stack.pop() != pairs[c]:\n                return False\n    return not stack\n",
        mutants: &[
            ("counts brackets, ignoring nesting order", "def is_balanced(s):\n    return s.count('(') == s.count(')') and s.count('[') == s.count(']') and s.count('{') == s.count('}')\n"),
            ("forgets leftovers on the stack", "def is_balanced(s):\n    pairs = {')':'(', ']':'[', '}':'{'}\n    stack = []\n    for c in s:\n        if c in '([{':\n            stack.append(c)\n        elif c in pairs:\n            if not stack or stack.pop() != pairs[c]:\n                return False\n    return True\n"),
        ],
    },
    PyAdequacy {
        task: "two_sum",
        reference: "def two_sum(nums, target):\n    seen = {}\n    for i, n in enumerate(nums):\n        if target - n in seen:\n            return [seen[target - n], i]\n        seen[n] = i\n    return []\n",
        mutants: &[
            ("returns the values, not the indices", "def two_sum(nums, target):\n    seen = {}\n    for i, n in enumerate(nums):\n        if target - n in seen:\n            return [target - n, n]\n        seen[n] = i\n    return []\n"),
            ("lets one element pair with itself", "def two_sum(nums, target):\n    for i in range(len(nums)):\n        for j in range(len(nums)):\n            if nums[i] + nums[j] == target:\n                return [i, j]\n    return []\n"),
        ],
    },
    PyAdequacy {
        task: "merge_intervals",
        reference: "def merge_intervals(intervals):\n    if not intervals:\n        return []\n    out = []\n    for s, e in sorted(intervals):\n        if out and s <= out[-1][1]:\n            out[-1][1] = max(out[-1][1], e)\n        else:\n            out.append([s, e])\n    return out\n",
        mutants: &[
            ("does not sort first", "def merge_intervals(intervals):\n    if not intervals:\n        return []\n    out = []\n    for s, e in intervals:\n        if out and s <= out[-1][1]:\n            out[-1][1] = max(out[-1][1], e)\n        else:\n            out.append([s, e])\n    return out\n"),
            ("touching intervals are left separate", "def merge_intervals(intervals):\n    if not intervals:\n        return []\n    out = []\n    for s, e in sorted(intervals):\n        if out and s < out[-1][1]:\n            out[-1][1] = max(out[-1][1], e)\n        else:\n            out.append([s, e])\n    return out\n"),
        ],
    },
    PyAdequacy {
        task: "longest_common_subsequence",
        reference: "def lcs(a, b):\n    m = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]\n    for i in range(1, len(a) + 1):\n        for j in range(1, len(b) + 1):\n            if a[i-1] == b[j-1]:\n                m[i][j] = m[i-1][j-1] + 1\n            else:\n                m[i][j] = max(m[i-1][j], m[i][j-1])\n    return m[len(a)][len(b)]\n",
        mutants: &[
            // The single most common misreading of the task.
            ("computes the longest common SUBSTRING", "def lcs(a, b):\n    best = 0\n    m = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]\n    for i in range(1, len(a) + 1):\n        for j in range(1, len(b) + 1):\n            if a[i-1] == b[j-1]:\n                m[i][j] = m[i-1][j-1] + 1\n                best = max(best, m[i][j])\n    return best\n"),
            ("counts shared characters regardless of order", "def lcs(a, b):\n    return sum(min(a.count(c), b.count(c)) for c in set(a))\n"),
        ],
    },
    PyAdequacy {
        task: "int_to_roman",
        reference: "def int_to_roman(n):\n    vals = [(1000,'M'),(900,'CM'),(500,'D'),(400,'CD'),(100,'C'),(90,'XC'),(50,'L'),(40,'XL'),(10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')]\n    out = ''\n    for v, sym in vals:\n        while n >= v:\n            out += sym\n            n -= v\n    return out\n",
        mutants: &[
            ("no subtractive forms (4 becomes IIII)", "def int_to_roman(n):\n    vals = [(1000,'M'),(500,'D'),(100,'C'),(50,'L'),(10,'X'),(5,'V'),(1,'I')]\n    out = ''\n    for v, sym in vals:\n        while n >= v:\n            out += sym\n            n -= v\n    return out\n"),
            ("handles 4 and 9 but not 40/90/400/900", "def int_to_roman(n):\n    vals = [(1000,'M'),(500,'D'),(100,'C'),(50,'L'),(10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')]\n    out = ''\n    for v, sym in vals:\n        while n >= v:\n            out += sym\n            n -= v\n    return out\n"),
        ],
    },
    PyAdequacy {
        task: "flatten_nested_list",
        reference: "def flatten(xs):\n    out = []\n    for x in xs:\n        if isinstance(x, list):\n            out.extend(flatten(x))\n        else:\n            out.append(x)\n    return out\n",
        mutants: &[
            ("flattens only one level", "def flatten(xs):\n    out = []\n    for x in xs:\n        if isinstance(x, list):\n            out.extend(x)\n        else:\n            out.append(x)\n    return out\n"),
            ("drops the non-list elements at the top level", "def flatten(xs):\n    out = []\n    for x in xs:\n        if isinstance(x, list):\n            out.extend(flatten(x))\n    return out\n"),
        ],
    },
];

pub(crate) fn tests_python_adequacy() -> CategoryResult {
    println!(
        "\n{}",
        bold(&cyan(
            "▶ Python oracle adequacy (do OUR checkers accept correct code and reject wrong code?)"
        ))
    );
    let mut results = Vec::new();

    let Some(py) = crate::code_gen_bench::python_cmd_pub() else {
        println!(
            "  {} skipping — no python interpreter found",
            yellow("SKIP")
        );
        results.push(TestResult {
            name: "prerequisites".to_string(),
            passed: true,
            message: Some("Skipped — python not available".to_string()),
            duration_ms: 0.0,
            score: None,
            details: Vec::new(),
            skipped: true,
            slow: false,
        });
        return CategoryResult {
            name: "python_adequacy".to_string(),
            results,
        };
    };

    for entry in PY_ADEQUACY {
        let Some(checker) = crate::code_gen_bench::checker_for(entry.task) else {
            results.push(run_test(&format!("py-oracle: {}", entry.task), || {
                Err(format!(
                    "no task named '{}' in code_gen_bench — the audit and the \
                     benchmark have drifted apart",
                    entry.task
                ))
            }));
            continue;
        };

        // 1. The checker must accept a correct implementation.
        results.push(run_test(
            &format!("py-oracle accepts correct: {}", entry.task),
            || {
                let src = format!("{}\n{}", entry.reference, checker);
                match crate::code_gen_bench::run_python_pub(py, &src) {
                    Ok(true) => Ok(()),
                    Ok(false) => Err("the checker REJECTS a correct implementation — it \
                                      would score a capable model as a failure"
                        .to_string()),
                    Err(e) => Err(format!("could not run python: {e}")),
                }
            },
        ));

        // 2. And reject every plausible wrong one.
        for (label, mutant) in entry.mutants {
            results.push(run_test(
                &format!("py-oracle catches '{}': {}", label, entry.task),
                || {
                    let src = format!("{}\n{}", mutant, checker);
                    match crate::code_gen_bench::run_python_pub(py, &src) {
                        Ok(false) => Ok(()),
                        Ok(true) => Err(format!(
                            "the checker ACCEPTS a wrong implementation ({label}) — every \
                             score it has ever produced was partly noise"
                        )),
                        Err(e) => Err(format!("could not run python: {e}")),
                    }
                },
            ));
        }
    }

    let passed = results.iter().filter(|r| r.passed && !r.skipped).count();
    let total = results.iter().filter(|r| !r.skipped).count();
    println!(
        "  {} python_adequacy: {}/{} oracle checks over {} tasks",
        bold(&cyan("∑")),
        passed,
        total,
        PY_ADEQUACY.len()
    );

    CategoryResult {
        name: "python_adequacy".to_string(),
        results,
    }
}
