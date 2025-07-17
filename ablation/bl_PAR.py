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
      -k k          low frequency Bayes hack    = 0  # CHANGED (was 1)
      -K Kuts       max discretization zones    = 17
      -l leaf       min size of tree leaves     = 2
      -m m          low frequency Bayes hack    = 0  # CHANGED (was 2)
      -p p          distance formula exponent   = 2  
      -r rseed      random number seed          = 1234567891  
      -s start      where to begin              = 4  
      -S Stop       where to end                = 32  
      -t tiny       min size of leaves of tree  = 4
      -v var_smoothing_gnb      variance smoothing          = 1e-9  # NEW
      -V alpha_cnb      alpha for bayes smoothing   = 1.0  # NEW
      -x BIG_EPS    constant                    = 1e-30  # NEW
"""
import re,sys,math,time,random

rand  = random.random
one   = random.choice
some  = random.choices
BIG   = 1E32

#--------- --------- --------- --------- --------- --------- ------- -------
class o:
  __init__ = lambda i,**d: i.__dict__.update(**d)
  __repr__ = lambda i: i.__class__.__name__ + show(i.__dict__)

def Num(txt=" ", at=0):
  return o(it=Num, txt=txt, at=at, n=0, mu=0, sd=0, m2=0, hi=-BIG, lo=BIG, 
           rank=0, # used by the stats functions, ignored otherwise
           goal = 0 if str(txt)[-1]=="-" else 1)

def Sym(txt=" ", at=0):
  return o(it=Sym, txt=txt, at=at, n=0, has={})

def Cols(names):
  cols = o(it=Cols, x=[], y=[], klass=-1, all=[], names=names)
  for n,s in enumerate(names):
    col = (Num if first(s).isupper() else Sym)(s,n)
    cols.all += [col]
    if s[-1] != "X":
      (cols.y if s[-1] in "+-!" else cols.x).append(col)
      if s[-1] == "!": cols.klass = col
  return cols

def Data(src=[]): return adds(src, o(it=Data,n=0,rows=[],cols=None))

def clone(data, src=[]): return adds(src, Data([data.cols.names]))

#--------- --------- --------- --------- --------- --------- ------- -------
def adds(src, i=None):
  for x in src:
    i = i or (Num() if isNum(x) else Sym())
    add(x,i)
  return i

def sub(v,i,  n=1): return add(v,i,n=n,flip=-1)

def add(v,i,  n=1,flip=1): # n only used for fast sym add
  def _sym(): 
    i.has[v] = flip * n  + i.has.get(v,0)
  def _data(): 
    if not i.cols: i.cols = Cols(v)  # called on first row
    elif flip < 0:# row subtraction managed elsewhere; e.g. see eg_addSub  
       [sub(v[col.at],col,n) for col in i.cols.all]  
    else:
       i.rows += [[add( v[col.at], col,n) for col in i.cols.all]]
  def _num():
    i.lo  = min(v, i.lo)
    i.hi  = max(v, i.hi)
    if flip < 0 and i.n < 2: 
      i.mu = i.sd = 0
    else:
      d     = v - i.mu
      i.mu += flip * (d / i.n)
      i.m2 += flip * (d * (v -   i.mu))
      i.sd  = 0 if i.n <=2  else (max(0,i.m2)/(i.n-1))**.5

  if v != "?":
    i.n += flip * n 
    _sym() if i.it is Sym else (_num() if i.it is Num else _data())
  return v

#--------- --------- --------- --------- --------- --------- ------- -------
def norm(v, col):
   if v=="?" or col.it is Sym: return v
   return (v - col.lo) / (col.hi - col.lo + 1/BIG)

def mid(col): 
  return col.mu if col.it is Num else max(col.has,key=col.has.get)

def spread(c): 
  if c.it is Num: return c.sd
  return -sum(n/c.n * math.log(n/c.n,2) for n in c.has.values() if n > 0)

def ydist(row,  data):
  return (sum(abs(norm(row[c.at], c) - c.goal)**the.p for c in data.cols.y) 
          / len(data.cols.y)) ** (1/the.p)

def ydists(rows, data): return sorted(rows, key=lambda row: ydist(row,data))

def yNums(rows,data): return adds(ydist(row,data) for row in rows)

#--------- --------- --------- --------- --------- --------- ------- -------
def likes(lst, datas):
  n = sum(data.n for data in datas)
  return max(datas, key=lambda data: like(lst, data, n, len(datas)))

def like(row, data, nall=100, nh=2):
  def _col(v,col): 
    if col.it is Sym: 
      return (col.has.get(v,0) + the.m*prior) / (col.n + the.m + 1/BIG)
    sd    = col.sd + 1/BIG
    nom   = math.exp(-1*(v - col.mu)**2/(2*sd*sd))
    denom = (2*math.pi*sd*sd) ** 0.5
    return max(0, min(1, nom/denom))

  prior = (data.n + the.k) / (nall + the.k*nh)
  tmp   = [_col(row[x.at], x) for x in data.cols.x if row[x.at] != "?"]
  return sum(math.log(n) for n in tmp + [prior] if n>0)

#--------- --------- --------- --------- --------- --------- ------- -------
def actLearn(data, shuffle=True):
  def _guess(row): 
    return _acquire(n/the.Stop, like(row,best,n,2), like(row,rest,n,2))
  def _acquire(p, b,r): 
    b,r = math.e**b, math.e**r
    q = 0 if the.acq=="xploit" else (1 if the.acq=="xplore" else 1-p)
    return (b + r*q) / abs(b*q - r + 1/BIG) 

  if shuffle: random.shuffle(data.rows)
  n     =  the.start
  todo  =  data.rows[n:]
  br    = clone(data, data.rows[:n])
  done  =  ydists(data.rows[:n], br)
  cut   =  round(n**the.guess)
  best  =  clone(data, done[:cut])
  rest  =  clone(data, done[cut:])
  while len(todo) > 2  and n < the.Stop:
    n      += 1
    hi, *lo = sorted(todo[:the.Few*2], key=_guess, reverse=True)
    todo    = lo[:the.Few] + todo[the.Few*2:] + lo[the.Few:]
    add(hi, best)
    add(hi, br)
    best.rows = ydists(best.rows, br)
    if len(best.rows) >= round(n**the.guess):
      add( sub(best.rows.pop(-1), best), rest)
  return o(best=best, rest=rest, todo=todo)

#--------- --------- --------- --------- --------- --------- ------- -------
def cuts(rows, col,Y,Klass=Num):
  def _v(row) : return row[col.at]
  def _upto(x): return f"{col.txt} <= {x} ", lambda z:_v(z)=="?" or _v(z)<=x
  def _over(x): return f"{col.txt} >  {x} ", lambda z:_v(z)=="?" or _v(z)>x
  def _eq(x)  : return f"{col.txt} == {x} ", lambda z:_v(z)=="?" or _v(z)==x
  def _sym():
    n,d = 0,{}
    for row in rows:
      x = _v(row) 
      if x != "?":
        d[x] = d.get(x) or Klass()
        add(Y(row), d[x])
        n = n + 1
    return o(entropy= sum(v.n/n * spread(v) for v in d.values()),
            decisions= [_eq(k) for k,v in d.items()])

  def _num():
    out,b4 = None,None 
    lhs, rhs = Klass(), Klass()
    xys = [(_v(r), add(Y(r),rhs)) for r in rows if _v(r) != "?"]
    xpect = spread(rhs)
    for x,y in sorted(xys, key=lambda xy: first(xy)):
      if the.leaf <= lhs.n <= len(xys) - the.leaf: 
        if x != b4:
          tmp = (lhs.n * spread(lhs) + rhs.n * spread(rhs)) / len(xys)
          if tmp < xpect:
            xpect, out = tmp,[_upto(b4), _over(b4)]
      add(sub(y, rhs),lhs)
      b4 = x
    if out:
      return o(entropy=xpect, decisions=out)

  return _sym() if col.it is Sym else _num()

#--------- --------- --------- --------- --------- --------- ------- -------
def tree(rows,data,Klass=Num,xplain="",decision=lambda _:True):
   def Y(row): return ydist(row,data)
   node        = clone(data,rows)
   node.ys     = yNums(rows,data).mu
   node.kids   = []
   node.decision  = decision
   node.xplain = xplain
   if len(rows) >= the.leaf:
     splits=[]
     for col in data.cols.x:
       if tmp := cuts(rows,col,Y,Klass=Klass): splits += [tmp]
     if splits:
       for xplain,decision in sorted(splits, key=lambda cut:cut.entropy)[0].decisions:
         rows1= [row for row in rows if decision(row)]
         if the.leaf <= len(rows1) < len(rows):
           node.kids += [tree(rows1,data,Klass=Klass,xplain=xplain,decision=decision)]
   return node   

def nodes(node,lvl=0, key=None):
  yield lvl,node
  for kid in (sorted(node.kids, key=key) if key else node.kids):
    for node1 in nodes(kid, lvl+1, key=key):
      yield node1

def showTree(tree, key=lambda z:z.ys):
  stats = yNums(tree.rows,tree)
  win = lambda x: 100-int(100*(x-stats.lo)/(stats.mu - stats.lo))
  print(f"{'d2h':>4} {'win':>4} {'n':>4}  ")
  print(f"{'----':>4} {'----':>4} {'----':>4}  ")
  for lvl, node in nodes(tree,key=key):
    leafp = len(node.kids)==0
    post= ";" if leafp else ""
    print(f"{node.ys:4.2f} {win(node.ys):4} {len(node.rows):4}    {(lvl-1) * '|  '}{node.xplain}" + post)

def leaf(node, row):
  for kid in node.kids or []:
    if kid.decision(row): 
      return leaf(kid,row)
  return node 
    
#--------- --------- --------- --------- --------- --------- ------- -------
def delta(i,j): 
  return abs(i.mu - j.mu) / ((i.sd**2/i.n + j.sd**2/j.n)**.5 + 1/BIG)

# non-parametric significance test From Introduction to Bootstrap, 
# Efron and Tibshirani, 1993, chapter 20. https://doi.org/10.1201/9780429246593"""
def bootstrap(vals1, vals2):
    x,y,z = adds(vals1+vals2), adds(vals1), adds(vals2)
    yhat  = [y1 - mid(y) + mid(x) for y1 in vals1]
    zhat  = [z1 - mid(z) + mid(x) for z1 in vals2] 
    n     = 0
    for _ in range(the.bootstraps):
      n += delta(adds(some(yhat,k=len(yhat))), 
                 adds(some(zhat,k=len(zhat)))) > delta(y,z) 
    return n / the.bootstraps >= (1- the.BootConf)

# Non-parametric effect size. Threshold is border between small=.11 and medium=.28 
# from Table1 of  https://doi.org/10.3102/10769986025002101
def cliffs(vals1,vals2):
   n,lt,gt = 0,0,0
   for x in vals1:
     for y in vals2:
        n += 1
        if x > y: gt += 1
        if x < y: lt += 1 
   return abs(lt - gt)/n  < the.cliffConf # 0.197) 

def vals2RankedNums(d, eps=0, reverse=False):
  def _samples():            return [_sample(d[k],k) for k in d]
  def _sample(vals,txt=" "): return o(vals=vals, num=adds(vals,Num(txt=txt)))
  def _same(b4,now):         return (abs(b4.num.mu - now.num.mu) < eps or
                                    cliffs(b4.vals, now.vals) and 
                                    bootstrap(b4.vals, now.vals))
  tmp,out = [],{}
  for now in sorted(_samples(), key=lambda z:z.num.mu, reverse=reverse):
    if tmp and _same(tmp[-1], now): 
      tmp[-1] = _sample(tmp[-1].vals + now.vals)
    else: 
      tmp += [ _sample(now.vals) ]
    now.num.rank = chr(96+len(tmp))
    out[now.num.txt] = now.num 
  return out

#--------- --------- --------- --------- --------- --------- ------- -------
def isNum(x): return isinstance(x,(float,int))

def first(lst): return lst[0] 

def coerce(s):
  try: return int(s)
  except Exception:
    try: return float(s)
    except Exception:
      s = s.strip()
      return True if s=="True" else (False if s=="False" else s)

def csv(file):
  with open(sys.stdin if file=="-" else file, encoding="utf-8") as src:
    for line in src:
      line = re.sub(r'([\n\t\r ]|#.*)', '', line)
      if line: yield [coerce(s) for s in line.split(",")]

def cli(d):
  for k,v in d.items():
    for c,arg in enumerate(sys.argv):
      if arg == "-"+first(k): 
        new = sys.argv[c+1] if c < len(sys.argv) - 1 else str(v)
        d[k] = coerce("False" if str(v) == "True"  else (
                      "True"  if str(v) == "False" else new))

def showd(x): print(show(x)); return x

def show(x):
  it = type(x)
  if   it is str   : x= f'"{x}"'
  elif callable(x) : x= x.__name__ + '()'
  elif it is float : x= str(round(x,the.decs))
  elif it is list  : x= '['+', '.join([show(v) for v in x])+']'
  elif it is dict  : x= "{"+' '.join([f":{k} {show(v)}" 
                          for k,v in x.items() if first(str(k)) !="_"]) +"}"
  return str(x)

def main():
  cli(the.__dict__)
  for n,s in enumerate(sys.argv):
    if fun := globals().get("eg" + s.replace("-","_")):
      arg = "" if n==len(sys.argv) - 1 else sys.argv[n+1]
      random.seed(the.rseed)
      fun(coerce(arg))

#--------- --------- --------- --------- --------- --------- ------- -------

def eg__nbfew(file, experiment_name="before_changes"):
    data = Data(csv(file or the.file))

    # Get the class column index
    class_col_idx = data.cols.klass.at

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

    # Define all possible n_pos values
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

    # Run experiments for different numbers of positive samples
    for n_pos in n_pos_values:
        recalls = []
        precisions = []
        f1s = []

        # Run 100 experiments for each n_pos
        for i in range(100):
            # Select positive samples
            selected_pos = random.sample(positive_samples, n_pos)

            # Select negative samples (4x the number of positive samples)
            n_neg = n_pos * 4
            if len(negative_samples) < n_neg:
                print(f"\nError: Not enough negative samples for n_pos={n_pos}.")
                print(f"Required: {n_neg} negative samples")
                print(f"Available: {len(negative_samples)} negative samples")
                return

            selected_neg = random.sample(negative_samples, n_neg)

            # Create separate datasets for positive and negative classes
            pos_dataset = clone(data, selected_pos)
            neg_dataset = clone(data, selected_neg)

            # Get remaining samples for testing
            test_samples = [
                row for row in data.rows if row not in selected_pos + selected_neg
            ]

            # Initialize counters for evaluation
            tp = 0  # True positives
            fp = 0  # False positives
            fn = 0  # False negatives
            tn = 0  # True negatives

            # Evaluate on test samples
            for row in test_samples:
                actual_class = row[class_col_idx]

                # Get the dataset with maximum likelihood
                best_dataset = likes(row, [pos_dataset, neg_dataset])

                # Predict positive class if the best dataset is the positive dataset
                predicted_class = (
                    positive_value
                    if best_dataset is pos_dataset
                    else (0 if is_numeric else "no")
                )

                # Update counters
                if actual_class == positive_value and predicted_class == positive_value:
                    tp += 1
                elif actual_class == positive_value and predicted_class != positive_value:
                    fn += 1
                elif actual_class != positive_value and predicted_class == positive_value:
                    fp += 1
                else:  # actual_class != positive_value and predicted_class != positive_value
                    tn += 1

            # Calculate precision, recall, and F1
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            recalls.append(recall)
            precisions.append(precision)
            f1s.append(f1)

            # Print recall and precision for ablation study parsing
            print(f"Recall: {recall:.3f}")
            print(f"Precision: {precision:.3f}")

            # Show progress every 10 experiments
            if (i + 1) % 10 == 0:
                print(f"Completed {i + 1} experiments for {n_pos} positive samples...")

        # Calculate statistics
        recalls = [r for r in recalls if r is not None]  # Remove any None values
        precisions = [p for p in precisions if p is not None]
        f1s = [f for f in f1s if f is not None]
        if not recalls:  # If all recalls are None, skip this iteration
            print(f"\nNo valid recalls for {n_pos} positive samples")
            continue

        # Sort for percentile calculations
        sorted_recalls = sorted(recalls)
        sorted_precisions = sorted(precisions)
        sorted_f1s = sorted(f1s)
        median = sorted_recalls[len(sorted_recalls) // 2]  # 50th percentile
        q1 = sorted_recalls[len(sorted_recalls) // 4]  # 25th percentile
        q3 = sorted_recalls[3 * len(sorted_recalls) // 4]  # 75th percentile
        median_p = sorted_precisions[len(sorted_precisions) // 2] if sorted_precisions else None
        q1_p = sorted_precisions[len(sorted_precisions) // 4] if sorted_precisions else None
        q3_p = sorted_precisions[3 * len(sorted_precisions) // 4] if sorted_precisions else None
        median_f1 = sorted_f1s[len(sorted_f1s) // 2] if sorted_f1s else None
        q1_f1 = sorted_f1s[len(sorted_f1s) // 4] if sorted_f1s else None
        q3_f1 = sorted_f1s[3 * len(sorted_f1s) // 4] if sorted_f1s else None

        print(f"\nResults for {n_pos} positive samples (100 experiments):")
        print(f"Recall: median={median:.3f}, Q1={q1:.3f}, Q3={q3:.3f}")
        if median_p is not None:
            print(f"Precision: median={median_p:.3f}, Q1={q1_p:.3f}, Q3={q3_p:.3f}")
        if median_f1 is not None:
            print(f"F1: median={median_f1:.3f}, Q1={q1_f1:.3f}, Q3={q3_f1:.3f}")

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

        # Store results for CSV output
        results["median"].append(str(median))
        results["q1"].append(str(q1))
        results["q3"].append(str(q3))

    # Save results to CSV
    csv_filename = f"results_{experiment_name}_before_changes.csv"
    with open(csv_filename, "w") as f:
        # Write header row with n_pos values
        f.write(",".join(map(str, n_pos_values)) + "\n")
        # Write data rows in order: median, q1, q3
        f.write(",".join(results["median"]) + "\n")
        f.write(",".join(results["q1"]) + "\n")
        f.write(",".join(results["q3"]) + "\n")

    print(f"\nResults saved to {csv_filename}")

#--------- --------- --------- --------- --------- --------- ------- -------
regx = r"-\w+\s*(\w+).*=\s*(\S+)"
the  = o(**{m[1]:coerce(m[2]) for m in re.finditer(regx,__doc__)})
random.seed(the.rseed)

if __name__ == "__main__":  main() 