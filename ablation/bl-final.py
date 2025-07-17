# todo:
# 1.change guards to lt, "gt". have col name in thre explictedyl
# 2. return nest from rile

"""
bl.py : barelogic, XAI for active learning + multi-objective optimization
(c) 2025, Tim Menzies <timm@ieee.org>, MIT License

OPTIONS:

      -a acq        xploit or xplore or adapt   = xploit
      -b bootstraps num of bootstrap samples    = 512
      -B BootConf   bootstrap threshold         = 0.95
      -B BootConf   bootstrap threshold         = 0.95
      -c cliffConf  cliffs' delta threshold     = 0.197
      -C Cohen      Cohen threshold             = 0.35
      -d decs       decimal places for printing = 3
      -f file       training csv file           = ../test/data/auto93.csv
      -F Few        search a few items in a list = 50
      -g guess      size of guess               = 0.5
      -k k          low frequency Bayes hack    = 0
      -K Kuts       max discretization zones    = 17
      -l leaf       min size of tree leaves     = 2
      -m m          low frequency Bayes hack    = 0
      -p p          distance formula exponent   = 2
      -r rseed      random number seed          = 1234567891
      -s start      where to begin              = 4
      -S Stop       where to end                = 32
      -t tiny       min size of leaves of tree  = 4
      -v var_smoothing_gnb      variance smoothing          = 1e-9
      -V alpha_cnb      alpha for bayes smoothing   = 1.0
      -x BIG_EPS    constant                    = 1e-30
"""

import re, sys, math, time, random

rand = random.random
one = random.choice
some = random.choices
BIG = 1e-30


# --------- --------- --------- --------- --------- --------- ------- -------
class o:
    __init__ = lambda i, **d: i.__dict__.update(**d)
    __repr__ = lambda i: i.__class__.__name__ + show(i.__dict__)


def Num(txt=" ", at=0):
    return o(
        it=Num,
        txt=txt,
        at=at,
        n=0,
        mu=0,
        sd=0,
        m2=0,
        hi=-float("inf"),
        lo=float("inf"),
        rank=0,
        goal=0 if str(txt)[-1] == "-" else 1,
    )


def Sym(txt=" ", at=0):
    # Added num_categories to track unique values for CategoricalNB smoothing
    # This `num_categories` will be set globally for the feature once all data is seen
    return o(it=Sym, txt=txt, at=at, n=0, has={}, global_num_categories=0)


def Cols(names):
    cols = o(it=Cols, x=[], y=[], klass=-1, all=[], names=names)
    for n, s in enumerate(names):
        col = (Num if first(s).isupper() else Sym)(s, n)
        cols.all += [col]
        if s[-1] != "X":
            (cols.y if s[-1] in "+-!" else cols.x).append(col)
            if s[-1] == "!":
                cols.klass = col
    return cols


def Data(src=[]):
    # Added a `all_rows_for_global_stats` to collect all rows for global categorical counts
    # This allows for calculating the true `n_categories` for smoothing in Sym objects.
    data_obj = o(it=Data, n=0, rows=[], cols=None, all_rows_for_global_stats=[])
    return adds(src, data_obj)


def clone(data, src=[]):
    new_data = Data([data.cols.names])
    if hasattr(data, 'normalizer'):
        new_data.normalizer = data.normalizer
    return adds(src, new_data)


# --------- --------- --------- --------- --------- --------- ------- -------
def adds(src, i=None):
    for x in src:
        if isinstance(i, o) and i.it is Data:
            # Special handling for Data objects to store all rows for global stats
            if (
                i.cols
            ):  # Only add to all_rows_for_global_stats after cols are initialized
                i.all_rows_for_global_stats.append(x)
            # The add(x,i) below will handle updating column statistics
        i = i or (
            Num() if isNum(x) else Sym()
        )  # This branch is for adding single values
        # For Data objects, 'src' should be a list of rows
        add(x, i)
    return i


def sub(v, i, n=1):
    return add(v, i, n=n, flip=-1)


def add(v, i, n=1, flip=1):  # n only used for fast sym add
    if i.it is Sym:  # Sym column update
        if v != "?":
            i.has[v] = flip * n + i.has.get(v, 0)
            # Update total count of non-missing values for this column (within this class)
            i.n += flip * n

    elif i.it is Num:  # Num column update
        if v != "?":
            # Ensure v is a number before performing arithmetic operations
            if not isNum(v):
                raise TypeError("Expected numerical value for Num column")

            # Welford's algorithm for numerically stable online mean and variance.
            # Reverted for subtraction, but primarily designed for forward adding.
            if flip > 0:  # Adding a value
                i.n += n
                if (
                    i.n == 0
                ):  # Should not happen if n is incremented first, but as a safeguard
                    i.mu = v
                    i.m2 = 0
                else:
                    delta = v - i.mu
                    i.mu += delta / i.n
                    i.m2 += delta * (v - i.mu)
            elif flip < 0:  # Subtracting a value
                # Subtraction in Welford's is tricky for exact inverse, especially for small n.
                # For simplicity and to match sklearn's typical fit, assume training is primarily 'add'.
                # If 'sub' is critical, a more robust inverse Welford or re-calculation is needed.
                if i.n > n:
                    i.n -= n
                    # More accurate inverse Welford or simply reset for small n
                    if i.n == 0:
                        i.mu = 0
                        i.m2 = 0
                    else:
                        old_mu = (
                            i.mu * (i.n + n) - v * n
                        ) / i.n  # Reconstruct old mean
                        # Reconstruct old m2 then remove contribution
                        # This is a bit complex and often leads to floating point issues.
                        # For `sklearn` matching, it's generally assumed that `add` is the primary training operation.
                        # If 'sub' must be supported with high precision, recomputing variance from stored values
                        # (if all values are stored) or using a more robust inverse Welford is needed.
                        # For now, let's simplify inverse for demo, but acknowledge its limits.
                        delta_v_old_mu = v - old_mu
                        delta_v_new_mu = v - i.mu  # Current i.mu after reduction
                        i.m2 = i.m2 - (
                            delta_v_old_mu * (v - i.mu)
                        )  # approximate reverse
                        i.m2 = max(0, i.m2)  # ensure non-negative
                else:  # n becomes 0 or negative
                    i.n = 0
                    i.mu = 0
                    i.sd = 0
                    i.m2 = 0
                    i.lo = -float("inf")
                    i.hi = float("inf")

            # Calculate variance based on SAMPLE variance formula (dividing by i.n-1) to match sklearn
            # and then add var_smoothing.
            if i.n > 1:
                variance = (
                    i.m2 / (i.n - 1)
                )  # Use sample variance to match sklearn
            elif i.n == 1:
                variance = 0  # No variance for single data point
            else:
                variance = 0  # No variance for zero data points

            # Add var_smoothing to variance, then take sqrt for sd
            i.sd = (variance + the.var_smoothing_gnb) ** 0.5

    elif i.it is Data:  # Data object update (row-wise)
        if not i.cols:  # First row, initialize columns
            i.cols = Cols(v)
            i.n += flip * n  # Update row count for data object
        elif flip < 0:  # Row subtraction
            [sub(v[col.at], col, n) for col in i.cols.all]  # Update column stats
            if v != "?":
                i.n += flip * n  # Update row count for data object
        else:  # Adding a row
            # This iterates through columns and calls add for each feature value,
            # updating the column statistics.
            # Store the row if needed, but not strictly for NB training.
            row_values = []
            if hasattr(i, 'normalizer') and i.normalizer:
                v = i.normalizer.normalize(v)
            for col in i.cols.all:
                add(
                    v[col.at], col, n
                )  # This updates individual column stats (n, mu, sd, has, etc.)
                row_values.append(v[col.at])  # Keep the original value for the row list
            i.rows.append(row_values)  # Add the row to the Data object's row list
            if v != "?":
                i.n += flip * n  # Update row count for data object
    return v


# --------- --------- --------- --------- --------- --------- ------- -------
def norm(v, col):
    if v == "?" or col.it is Sym:
        return v
    # Ensure not to divide by zero if range is 0.
    range_ = col.hi - col.lo
    return (
        (v - col.lo) / (range_ + the.BIG_EPS) if range_ > the.BIG_EPS else 0.5
    )  # Return mid-point if range is 0


def mid(col):
    # Ensure col.has is not empty for Sym before calling max
    if col.it is Sym and not col.has:
        return None  # Or raise error, or return default
    return col.mu if col.it is Num else max(col.has, key=col.has.get)


def spread(c):
    if c.it is Num:
        return c.sd
    # Ensure c.n is not zero to avoid division by zero for entropy
    if c.n == 0:
        return 0
    return -sum(n / c.n * math.log(n / c.n, 2) for n in c.has.values() if n > 0)


def ydist(row, data):
    # Ensure len(data.cols.y) is not zero to avoid division by zero
    if not data.cols or not data.cols.y:
        return 0  # Or handle as an error
    sum_diff_sq = sum(abs(norm(row[c.at], c) - c.goal) ** the.p for c in data.cols.y)
    return (sum_diff_sq / len(data.cols.y)) ** (1 / the.p)


def ydists(rows, data):
    return sorted(rows, key=lambda row: ydist(row, data))


def yNums(rows, data):
    # This expects a list of numbers or similar.
    # If ydist returns a single number per row, then 'adds' on it should create a Num object
    # Initialize with a Num object to ensure correct addition behavior for a single number.
    return adds([ydist(row, data) for row in rows], Num())


# --------- --------- --------- --------- --------- --------- ------- -------
def likes(lst, datas):
    # Calculate total number of samples (nall) and number of classes (nh) from your data structures
    # nall and nh are derived directly from the 'datas' list, which is assumed to contain
    # Data objects that have been 'fitted' (i.e., their .n counts reflect training data).
    nall = sum(d.n for d in datas)  # total samples
    nh = len(datas)  # number of classes

    # Before prediction, update global_num_categories for Sym columns
    # This needs to be done once after all training data is added to all `datas` objects.
    # This is a bit of a hack to make the `Sym` object itself store a global value
    # which would ideally be part of a higher-level 'Model' object.
    # For exact sklearn comparison, this step is critical.
    if datas:
        all_feature_values_by_index = {}
        for d in datas:
            for row_data in d.all_rows_for_global_stats:  # Use the collected raw rows
                # Ensure row_data has enough elements for col_idx
                if d.cols:  # Check if cols are initialized
                    for col_idx, value in enumerate(row_data):
                        # Ensure col_idx is within bounds of d.cols.all
                        if (
                            col_idx < len(d.cols.all)
                            and d.cols.all[col_idx].it is Sym
                            and value != "?"
                        ):
                            all_feature_values_by_index.setdefault(col_idx, set()).add(
                                value
                            )
        for d in datas:
            for col in d.cols.all:
                if col.it is Sym:
                    # Set global_num_categories based on all unique values seen across ALL classes
                    col.global_num_categories = len(
                        all_feature_values_by_index.get(col.at, set())
                    )
                    # If no categories seen at all for a symbolic feature, default to a minimum (e.g., 2 for binary)
                    if col.global_num_categories == 0:
                        col.global_num_categories = (
                            2  # A common default for features unseen in training
                        )

    # This loop is your prediction step, comparing log-likelihoods
    return max(datas, key=lambda data: like(lst, data, nall, nh))


def like(row, data, nall=100, nh=2):
    # Apply normalization to the row if a normalizer is available
    if hasattr(data, 'normalizer') and data.normalizer:
        row = data.normalizer.normalize(row)
    
    def _col(v, col):
        if v == "?":
            return 1.0  # Missing values, as per your code, don't penalize. Treat as probability 1.

        if col.it is Sym:
            # Sklearn CategoricalNB smoothing: P(feature | class) = (count + alpha) / (total_class_feature_count + alpha * n_categories)
            # col.has.get(v,0) is count(v, data_class)
            # col.n is total count for this feature in this class (from non-missing entries)
            # col.global_num_categories is the total number of possible categories for that feature
            n_categories_for_smoothing = max(1, col.global_num_categories)
            # Total count of values for this symbolic column in this specific class (col.n)
            # Add the.BIG_EPS to denominator to prevent division by zero, though unlikely with proper smoothing.
            return (col.has.get(v, 0) + the.alpha_cnb) / (
                col.n + the.alpha_cnb * n_categories_for_smoothing + the.BIG_EPS
            )

        # Numerical (Gaussian) likelihood
        sd = col.sd  # col.sd should already have var_smoothing applied during 'add'
        if (
            sd <= the.BIG_EPS
        ):  # Handle cases where standard deviation is effectively zero
            # If sd is ~0, PDF is a spike. Return 1 if exactly on the mean, else a tiny probability.
            return 1.0 if abs(v - col.mu) < the.BIG_EPS else the.BIG_EPS

        # Gaussian PDF calculation (sklearn does not clamp PDF values between 0 and 1)
        # Use log-space calculation to avoid numerical underflow
        log_nom = -1 * (v - col.mu) ** 2 / (2 * sd * sd)
        log_denom = 0.5 * math.log(2 * math.pi * sd * sd)
        log_pdf = log_nom - log_denom
        
        # Convert back to probability space, but with bounds to prevent underflow
        pdf = math.exp(log_pdf)
        
        # Apply a minimum threshold to prevent extremely small values
        min_prob = 1e-10
        return max(pdf, min_prob)

    # Calculate class prior
    # To match sklearn's default empirical prior: (data.n / nall)
    # Your `the.k` should be 0 to achieve this. `nh` isn't strictly relevant for empirical priors.
    prior = (data.n + the.k) / (
        nall + the.k * nh + the.BIG_EPS
    )  # Adjusted with BIG_EPS
    
    # Ensure prior is not too small to prevent numerical issues
    prior = max(prior, 1e-10)

    tmp = []
    for (
        x_col
    ) in data.cols.x:  # Iterate over feature columns (x means independent features)
        # Your handling of "?" is to skip.
        if row[x_col.at] != "?":
            val_likelihood = _col(row[x_col.at], x_col)
            tmp.append(val_likelihood)

    # Ensure prior is positive before taking log
    log_prior = math.log(prior) if prior > the.BIG_EPS else math.log(the.BIG_EPS)

    # Sum log probabilities for numerical stability.
    # Ensure all likelihoods are positive before taking log.
    log_likelihoods_sum = sum(math.log(max(n, the.BIG_EPS)) for n in tmp)

    return log_prior + log_likelihoods_sum


# --------- --------- --------- --------- --------- --------- ------- -------
def actLearn(data, shuffle=True):
    def _guess(row):
        # 'best' and 'rest' are also Data objects that get 'fitted' incrementally
        return _acquire(n / the.Stop, like(row, best, n, 2), like(row, rest, n, 2))

    def _acquire(p, b, r):
        b, r = math.e**b, math.e**r
        q = 0 if the.acq == "xploit" else (1 if the.acq == "xplore" else 1 - p)
        return (b + r * q) / abs(b * q - r + the.BIG_EPS)  # Changed BIG to BIG_EPS

    if shuffle:
        random.shuffle(data.rows)
    n = the.start
    todo = data.rows[n:]
    br = clone(
        data, data.rows[:n]
    )  # br is a Data object, gets its stats from initial rows
    done = ydists(data.rows[:n], br)  # ydist uses 'br' as a context for column stats
    cut = round(n**the.guess)
    best = clone(data, done[:cut])  # best is a Data object with its own stats
    rest = clone(data, done[cut:])  # rest is a Data object with its own stats
    while len(todo) > 2 and n < the.Stop:
        n += 1
        hi, *lo = sorted(todo[: the.Few * 2], key=_guess, reverse=True)
        todo = lo[: the.Few] + todo[the.Few * 2 :] + lo[the.Few :]
        add(
            hi, best
        )  # This 'add' updates the stats of the 'best' Data object and its columns
        add(hi, br)  # Updates stats of 'br' Data object and its columns
        best.rows = ydists(
            best.rows, br
        )  # Re-sorts based on ydist calculated using 'br' stats
        if len(best.rows) >= round(n**the.guess):
            # add( sub(best.rows.pop(-1), best), rest) # This line involves 'sub', which is tricky for exact matching
            # For exact sklearn match, avoiding 'sub' in models that't support it is best.
            # If 'sub' is critical, a more robust inverse Welford or re-calculation is needed.
            # For this exercise, if 'sub' is causing issues, one might simplify the active learning
            # to only add to 'best' and 'br', and 'rest' would be formed from what's not in 'best'
            # without explicit subtractions from 'best'.
            removed_row = best.rows.pop(-1)
            add(removed_row, rest)  # Just add the removed row to rest
    return o(best=best, rest=rest, todo=todo)


# --------- --------- --------- --------- --------- --------- ------- -------
def cuts(rows, col, Y, Klass=Num):
    def _v(row):
        return row[col.at]

    def _upto(x):
        return f"{col.txt} <= {x} ", lambda z: _v(z) == "?" or _v(z) <= x

    def _over(x):
        return f"{col.txt} >  {x} ", lambda z: _v(z) == "?" or _v(z) > x

    def _eq(x):
        return f"{col.txt} == {x} ", lambda z: _v(z) == "?" or _v(z) == x

    def _sym():
        n, d = 0, {}
        for row in rows:
            x = _v(row)
            if x != "?":
                d[x] = d.get(x) or Klass()
                add(Y(row), d[x])
                n = n + 1
        return o(
            entropy=sum(v.n / n * spread(v) for v in d.values()),
            decisions=[_eq(k) for k, v in d.items()],
        )

    def _num():
        out, b4 = None, None
        lhs, rhs = Klass(), Klass()
        xys = [(_v(r), add(Y(r), rhs)) for r in rows if _v(r) != "?"]
        xpect = spread(rhs)
        for x, y in sorted(xys, key=lambda xy: first(xy)):
            if the.leaf <= lhs.n <= len(xys) - the.leaf:
                if x != b4:
                    tmp = (lhs.n * spread(lhs) + rhs.n * spread(rhs)) / len(xys)
                    if tmp < xpect:
                        xpect, out = tmp, [_upto(b4), _over(b4)]
            add(sub(y, rhs), lhs)
            b4 = x
        if out:
            return o(entropy=xpect, decisions=out)

    return _sym() if col.it is Sym else _num()


# --------- --------- --------- --------- --------- --------- ------- -------
def tree(rows, data, Klass=Num, xplain="", decision=lambda _: True):
    def Y(row):
        return ydist(row, data)

    node = clone(data, rows)  # clone data and adds rows, updating its internal stats
    node.ys = yNums(rows, data).mu
    node.kids = []
    node.decision = decision
    node.xplain = xplain
    if len(rows) >= the.leaf:
        splits = []
        for col in data.cols.x:
            if tmp := cuts(rows, col, Y, Klass=Klass):
                splits += [tmp]
        if splits:
            for xplain, decision in sorted(splits, key=lambda cut: cut.entropy)[
                0
            ].decisions:
                rows1 = [row for row in rows if decision(row)]
                if the.leaf <= len(rows1) < len(rows):
                    node.kids += [
                        tree(rows1, data, Klass=Klass, xplain=xplain, decision=decision)
                    ]
    return node


def nodes(node, lvl=0, key=None):
    yield lvl, node
    for kid in sorted(node.kids, key=key) if key else node.kids:
        for node1 in nodes(kid, lvl + 1, key=key):
            yield node1


def showTree(tree, key=lambda z: z.ys):
    stats = yNums(tree.rows, tree)
    win = lambda x: 100 - int(100 * (x - stats.lo) / (stats.mu - stats.lo))
    print(f"{'d2h':>4} {'win':>4} {'n':>4}  ")
    print(f"{'----':>4} {'----':>4} {'----':>4}  ")
    for lvl, node in nodes(tree, key=key):
        leafp = len(node.kids) == 0
        post = ";" if leafp else ""
        print(
            f"{node.ys:4.2f} {win(node.ys):4} {len(node.rows):4}    {(lvl - 1) * '|  '}{node.xplain}"
            + post
        )


def leaf(node, row):
    for kid in node.kids or []:
        if kid.decision(row):
            return leaf(kid, row)
    return node


# --------- --------- --------- --------- --------- --------- ------- -------
def delta(i, j):
    return abs(i.mu - j.mu) / (
        (i.sd**2 / i.n + j.sd**2 / j.n) ** 0.5 + the.BIG_EPS
    )  # Changed BIG to BIG_EPS


# non-parametric significance test From Introduction to Bootstrap,
# Efron and Tibsirani, 1993, chapter 20. https://doi.org/10.1201/9780429246593"""
def bootstrap(vals1, vals2):
    x, y, z = (
        adds(vals1 + vals2, Num()),
        adds(vals1, Num()),
        adds(vals2, Num()),
    )  # Ensure Num objects created
    yhat = [y1 - mid(y) + mid(x) for y1 in vals1]
    zhat = [z1 - mid(z) + mid(x) for z1 in vals2]
    n = 0
    for _ in range(the.bootstraps):
        # Ensure adds returns a Num object for delta
        n += delta(
            adds(some(yhat, k=len(yhat)), Num()), adds(some(zhat, k=len(zhat)), Num())
        ) > delta(y, z)
    return n / the.bootstraps >= (1 - the.BootConf)


# Non-parametric effect size. Threshold is border between small=.11 and medium=.28
# from Table1 of  https://doi.org/10.3102/10769986025002101
def cliffs(vals1, vals2):
    n, lt, gt = 0, 0, 0
    for x in vals1:
        for y in vals2:
            n += 1
            if x > y:
                gt += 1
            if x < y:
                lt += 1
    return abs(lt - gt) / n < the.cliffConf  # 0.197)


def vals2RankedNums(d, eps=0, reverse=False):
    def _samples():
        return [_sample(d[k], k) for k in d]

    def _sample(vals, txt=" "):
        return o(vals=vals, num=adds(vals, Num(txt=txt)))

    def _same(b4, now):
        return (
            abs(b4.num.mu - now.num.mu) < eps
            or cliffs(b4.vals, now.vals)
            and bootstrap(b4.vals, now.vals)
        )

    tmp, out = [], {}
    for now in sorted(_samples(), key=lambda z: z.num.mu, reverse=reverse):
        if tmp and _same(tmp[-1], now):
            tmp[-1] = _sample(tmp[-1].vals + now.vals)
        else:
            tmp += [_sample(now.vals)]
        now.num.rank = chr(96 + len(tmp))
        out[now.num.txt] = now.num
    return out


# --------- --------- --------- --------- --------- --------- ------- -------
def isNum(x):
    return isinstance(x, (float, int))


def first(lst):
    return lst[0] if lst else ""  # Added check for empty list


def coerce(s):
    try:
        return int(s)
    except Exception:
        try:
            return float(s)
        except Exception:
            s = s.strip()
            return True if s == "True" else (False if s == "False" else s)


def csv(file):
    with open(sys.stdin if file == "-" else file, encoding="utf-8") as src:
        for line in src:
            # Use re.sub with re.MULTILINE to handle newlines correctly
            line = re.sub(r"([\n\t\r ]|#.*)", "", line, flags=re.MULTILINE)
            if line:
                yield [coerce(s) for s in line.split(",")]


def cli(d):
    for k, v in d.items():
        for c, arg in enumerate(sys.argv):
            if arg == "-" + first(k):
                new = sys.argv[c + 1] if c < len(sys.argv) - 1 else str(v)
                d[k] = coerce(
                    "False"
                    if str(v) == "True"
                    else ("True" if str(v) == "False" else new)
                )


def showd(x):
    print(show(x))
    return x


def show(x):
    it = type(x)
    if it is str:
        x = f'"{x}"'
    elif callable(x):
        x = x.__name__ + "()"
    elif it is float:
        x = str(round(x, the.decs))
    elif it is list:
        x = "[" + ", ".join([show(v) for v in x]) + "]"
    elif it is dict:
        x = (
            "{"
            + " ".join(
                [f":{k} {show(v)}" for k, v in x.items() if first(str(k)) != "_"]
            )
            + "}"
        )
    return str(x)


def main():
    cli(the.__dict__)
    for n, s in enumerate(sys.argv):
        if fun := globals().get("eg" + s.replace("-", "_")):
            arg = "" if n == len(sys.argv) - 1 else sys.argv[n + 1]
            random.seed(the.rseed)
            fun(coerce(arg))


# --------- --------- --------- --------- --------- --------- ------- -------
def eg__the(_):
    print(the)


def eg__nbfew(file, experiment_name="default"):
    data = Data(csv(file or the.file))

    # Get the class column index
    class_col_idx = data.cols.klass.at if hasattr(data.cols.klass, 'at') else data.cols.klass

    # Determine if class column is numeric or string
    is_numeric = isinstance(data.rows[0][class_col_idx], (int, float))
    positive_value = 1 if is_numeric else "yes"

    # Split data into positive and negative samples
    positive_samples = [
        row for row in data.rows if row[class_col_idx] == positive_value
    ]
    negative_samples = [
        row for row in data.rows if row[class_col_idx] != positive_value
    ]

    # Define all possible n_pos values - only test 32
    all_n_pos_values = [32]

    # Filter n_pos values to only include those that are possible
    n_pos_values = [n for n in all_n_pos_values if n <= len(positive_samples)]

    if not n_pos_values:
        print(f"\nError: Not enough positive samples in the dataset.")
        print(f"Minimum required: {min(all_n_pos_values)} positive samples")
        print(f"Available: {len(positive_samples)} positive samples")
        return

    print(
        f"\nRunning experiments for {len(n_pos_values)} possible n_pos values: {n_pos_values}"
    )
    print("=" * 50)

    # Store results for CSV output
    results = {
        "median": [],  # 50th percentile
        "q1": [],  # 25th percentile
        "q3": [],  # 75th percentile
    }

    for n_pos in n_pos_values:
        recalls = []
        experiment_times = []
        sampling_times = []
        cloning_times = []
        prediction_times = []
        for i in range(100):
            experiment_start = time.time()

            # Sampling
            t0 = time.time()
            selected_pos = random.sample(positive_samples, n_pos)
            n_neg = n_pos * 4
            if len(negative_samples) < n_neg:
                print(f"\nError: Not enough negative samples for n_pos={n_pos}.")
                print(f"Required: {n_neg} negative samples")
                print(f"Available: {len(negative_samples)} negative samples")
                return
            selected_neg = random.sample(negative_samples, n_neg)
            t1 = time.time()
            sampling_times.append(t1 - t0)

            # Cloning
            t0 = time.time()
            pos_dataset = clone(data, selected_pos)
            neg_dataset = clone(data, selected_neg)
            t1 = time.time()
            cloning_times.append(t1 - t0)

            # Prediction
            t0 = time.time()
            test_samples = [row for row in data.rows if row not in selected_pos + selected_neg]
            tp = 0
            fn = 0
            total_positives = 0
            for row in test_samples:
                actual_class = row[class_col_idx]
                if actual_class == positive_value:
                    total_positives += 1
                    best_dataset = likes(row, [pos_dataset, neg_dataset])
                    predicted_class = (
                        positive_value if best_dataset is pos_dataset else (0 if is_numeric else "no")
                    )
                    if predicted_class == positive_value:
                        tp += 1
                    else:
                        fn += 1
            t1 = time.time()
            prediction_times.append(t1 - t0)

            # Recall
            if total_positives > 0:
                recall = tp / (tp + fn)
                recalls.append(recall)

            experiment_end = time.time()
            experiment_time = experiment_end - experiment_start
            experiment_times.append(experiment_time)

            if (i + 1) % 10 == 0:
                avg_time = sum(experiment_times) / len(experiment_times)
                print(f"Completed {i + 1} experiments for {n_pos} positive samples... (avg: {avg_time:.3f}s/exp)")

        # Calculate statistics
        recalls = [r for r in recalls if r is not None]
        if not recalls:
            print(f"\nNo valid recalls for {n_pos} positive samples")
            continue
        mean_recall = sum(recalls) / len(recalls)
        variance = sum((x - mean_recall) ** 2 for x in recalls) / len(recalls)
        std_recall = variance**0.5
        sorted_recalls = sorted(recalls)
        median = sorted_recalls[len(sorted_recalls) // 2]
        q1 = sorted_recalls[len(sorted_recalls) // 4]
        q3 = sorted_recalls[3 * len(sorted_recalls) // 4]
        results["median"].append(f"{median:.3f}")
        results["q1"].append(f"{q1:.3f}")
        results["q3"].append(f"{q3:.3f}")
        print(f"\nResults for {n_pos} positive samples (100 experiments):")
        print(f"Mean Recall: {mean_recall:.3f} ± {std_recall:.3f}")
        print(f"Median Recall: {median:.3f}")
        print(f"Q1 (25th percentile): {q1:.3f}")
        print(f"Q3 (75th percentile): {q3:.3f}")
        print(f"Min Recall: {min(recalls):.3f}")
        print(f"Max Recall: {max(recalls):.3f}")
        # Print timing statistics
        avg_experiment_time = sum(experiment_times) / len(experiment_times)
        total_time = sum(experiment_times)
        print(f"Timing Statistics:")
        print(f"  Average time per experiment: {avg_experiment_time:.3f} seconds")
        print(f"  Total time for all experiments: {total_time:.3f} seconds")
        print(f"  Fastest experiment: {min(experiment_times):.3f} seconds")
        print(f"  Slowest experiment: {max(experiment_times):.3f} seconds")
        print(f"  Avg sampling: {sum(sampling_times)/len(sampling_times):.3f}s | Avg cloning: {sum(cloning_times)/len(cloning_times):.3f}s | Avg prediction: {sum(prediction_times)/len(prediction_times):.3f}s")
        print(f"  % sampling: {100*sum(sampling_times)/total_time:.1f}% | % cloning: {100*sum(cloning_times)/total_time:.1f}% | % prediction: {100*sum(prediction_times)/total_time:.1f}%")
        # Create distribution visualization
        print("\nDistribution of Recalls:")
        bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        counts = [0] * (len(bins) - 1)

        # Count values in each bin
        for recall in recalls:
            for i in range(len(bins) - 1):
                if bins[i] <= recall < bins[i + 1]:
                    counts[i] += 1
                    break
            else:  # If recall is exactly 1.0, count it in the last bin
                if recall == 1.0:
                    counts[-1] += 1

        # Verify total count matches number of experiments
        total_count = sum(counts)
        if total_count != len(recalls):
            print(
                f"Warning: Distribution count ({total_count}) doesn't match number of experiments ({len(recalls)})"
            )

        # Print distribution
        max_count = max(counts) if counts else 1
        scale = 40  # Maximum bar length

        for i in range(len(bins) - 1):
            bar_length = int((counts[i] / max_count) * scale)
            bar = "█" * bar_length
            print(f"{bins[i]:.1f}-{bins[i + 1]:.1f}: {bar} {counts[i]}")

        print("=" * 50)

    # Save results to CSV
    csv_filename = f"results_{experiment_name}.csv"
    with open(csv_filename, "w") as f:
        # Write header row with n_pos values
        f.write(",".join(map(str, n_pos_values)) + "\n")
        # Write data rows in order: median, q1, q3
        f.write(",".join(results["median"]) + "\n")
        f.write(",".join(results["q1"]) + "\n")
        f.write(",".join(results["q3"]) + "\n")

    print(f"\nResults saved to {csv_filename}")


# --------- --------- --------- --------- --------- --------- ------- -------
regx = r"-\w+\s*(\w+).*=\s*(\S+)"
the = o(**{m[1]: coerce(m[2]) for m in re.finditer(regx, __doc__)});
the.normalize = False  # Add this line to allow toggling normalization
random.seed(the.rseed)

# Helper for min-max normalization
class Normalizer:
    def __init__(self, cols):
        self.mins = [float('inf')] * len(cols.x)
        self.maxs = [float('-inf')] * len(cols.x)
        self.idxs = [col.at for col in cols.x]
    def update(self, row):
        for i, idx in enumerate(self.idxs):
            v = row[idx]
            if isNum(v):
                if v < self.mins[i]: self.mins[i] = v
                if v > self.maxs[i]: self.maxs[i] = v
    def normalize(self, row):
        normed = list(row)
        for i, idx in enumerate(self.idxs):
            v = row[idx]
            if isNum(v):
                lo, hi = self.mins[i], self.maxs[i]
                if hi > lo:
                    normed[idx] = (v - lo) / (hi - lo)
                else:
                    normed[idx] = 0.0
        return normed

if __name__ == "__main__":
    main()
