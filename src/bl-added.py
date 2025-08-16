
# todo: 
# 1.change guards to lt, "gt". have col name in thre explictedyl
# 2. return nest from rile

"""
bl-added.py : barelogic, XAI for active learning + multi-objective optimization
(c) 2025, Tim Menzies <timm@ieee.org>, MIT License  

OPTIONS:  

      -a acq        xploit or xplore or adapt   = xploit  
      -b bootstraps num of bootstrap samples    = 512
      -B BootConf   bootstrap threshold         = 0.95
      -c cliffConf  cliffs' delta threshold     = 0.197
      -C Cohen      Cohen threshold             = 0.35
      -d decs       decimal places for printing = 3  
      -f file       training csv file           = ../test/data/auto93.csv  
      -F Few        search a few items in a list = 50
      -g guess      size of guess               = 0.5  
      -k k          low frequency Bayes hack    = 1  
      -K Kuts       max discretization zones    = 17
      -l leaf       min size of tree leaves     = 2
      -m m          low frequency Bayes hack    = 2  
      -p p          distance formula exponent   = 2  
      -r rseed      random number seed          = 1234567891  
      -s start      where to begin              = 4  
      -S Stop       where to end                = 32  
      -t tiny       min size of leaves of tree  = 4 
      -M Mode       bl or sklearn      = bl 
      -v var_smoothing_gnb_sk      variance smoothing (sklearn mode)         = 1e-9 
      -A Alpha_cnb_sk      alpha for bayes smoothing (sklearn mode)   = 1.0 
      -e eps_sk    constant (sklearn mode)                    = 1e-30 
      -n n_pos     number of positive examples (sklearn mode) = 8
"""
import re,sys,math,time,random,os
import copy
import concurrent.futures

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
  return o(it=Sym, txt=txt, at=at, n=0, has={}, global_num_categories=0)

def Cols(names):
  cols = o(it=Cols, x=[], y=[], klass=-1, all=[], names=names)
  for n,s in enumerate(names):
    col = (Num if first(s).isupper() else Sym)(s,n)
    cols.all += [col]
    if s[-1] != "X":
      (cols.y if s[-1] in "+-!" else cols.x).append(col)
      if s[-1] == "!": cols.klass = col
  return cols

def Data(src=[]): return adds(src, o(it=Data,n=0,rows=[],cols=None,all_rows_for_global_stats=[]))

def clone(data, src=[]): return adds(src, Data([data.cols.names]))

#--------- --------- --------- --------- --------- --------- ------- -------
def adds(src, i=None):
  for x in src:
    if isinstance(i, o) and getattr(i, 'it', None) is Data:
      if i.cols:
        i.all_rows_for_global_stats.append(x)
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

#--------- --------- like/likes wrapper --------- ------- -------
def likes(row, datas):
    if the.Mode == 'sklearn':
        # Assume global category stats are already calculated
        return likes_sklearn(row, datas)
    elif the.Mode == 'bl':
        return likes_traditional(row, datas)
    else:
        raise ValueError(f"Unknown mode: {the.Mode}")

def like(row, data, nall=100, nh=2):
    if the.Mode == 'sklearn':
        return like_sklearn(row, data, nall, nh)
    elif the.Mode == 'bl':
        return like_traditional(row, data, nall, nh)
    else:
        raise ValueError(f"Unknown mode: {the.Mode}")
    
#--------- --------- bl mode likes/like logic --------- ------- -------
def likes_traditional(lst, datas):
    n = sum(data.n for data in datas)
    return max(datas, key=lambda data: like_traditional(lst, data, n, len(datas)))

def like_traditional(row, data, nall=100, nh=2):
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

#--------- --------- sklearn mode likes/like logic --------- ------- -------
def calculate_global_category_stats(datas):
    if not datas:
        return
    all_feature_values_by_index = {}
    for d in datas:
        for row_data in getattr(d, 'all_rows_for_global_stats', []):
            if getattr(d, 'cols', None):
                for col_idx, value in enumerate(row_data):
                    if (
                        col_idx < len(d.cols.all)
                        and getattr(d.cols.all[col_idx], 'it', None) is Sym
                        and value != "?"
                    ):
                        all_feature_values_by_index.setdefault(col_idx, set()).add(value)
    for d in datas:
        for col in getattr(d, 'cols', []).all if getattr(d, 'cols', None) else []:
            if getattr(col, 'it', None) is Sym:
                col.global_num_categories = len(all_feature_values_by_index.get(col.at, set()))
                if col.global_num_categories == 0:
                    col.global_num_categories = 2

def likes_sklearn(lst, datas):
    nall = sum(d.n for d in datas)  # total samples
    nh = len(datas)  # number of classes
    return max(datas, key=lambda data: like_sklearn(lst, data, nall, nh))

def _log_pdf_for_col(value, col):
   if value == "?":
      return 0.0
   
   # Handle symbolic columns
   if getattr(col, 'it', None) is Sym:
      n_categories_for_smoothing = max(1, getattr(col, 'global_num_categories', 1))
      return (col.has.get(value, 0) + the.Alpha_cnb_sk) / (
          col.n + the.Alpha_cnb_sk * n_categories_for_smoothing + the.eps_sk)
   
   # Handle numeric columns
   if getattr(col, 'it', None) is Num:
      mu = getattr(col, 'mu', 0)
      sd = getattr(col, 'sd', 1)
      
      variance = sd * sd + the.var_smoothing_gnb_sk
      log_nom = -1 * (value - mu) ** 2 / (2 * variance)
      log_denom = 0.5 * math.log(2 * math.pi * variance)
      
      return log_nom - log_denom
   
   # Default case - return 0 if column type is unknown
   return 0.0

def like_sklearn(row, data, nall=100, nh=2):
    prior = (data.n) / (nall + the.eps_sk)
    log_prior = math.log(max(prior, the.eps_sk))

    total_log_likelihood = sum(
        _log_pdf_for_col(row[x_col.at], x_col)
        for x_col in getattr(data, 'cols', []).x if getattr(data, 'cols', None)
    )

    return log_prior + total_log_likelihood

#--------- --------- --------- --------- --------- --------- ------- -------
def actLearn(data, shuffle=True):
    def _acquire(p, b, r): 
        b, r = math.e**b, math.e**r
        q = 0 if the.acq == "xploit" else (1 if the.acq == "xplore" else 1-p)
        return (b + r*q) / abs(b*q - r + 1/BIG) 

    def _guess(row): 
        return _acquire(n/the.Stop, like(row, best, n, 2), like(row, rest, n, 2))

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
            add(sub(best.rows.pop(-1), best), rest)
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
def eg__the(_): print(the)

def eg__cols(_): 
  s="Clndrs,Volume,HpX,Model,origin,Lbs-,Acc+,Mpg+"
  [print(col) for col in Cols(s.split(",")).all]

def eg__csv(file): 
  rows =list(csv(file or the.file))
  assert 3192 == sum(len(row) for row in rows)
  for row in rows[1:]: assert type(first(row)) is int

def eg__data(file):
  data=Data(csv(file or the.file))
  assert 3184 == sum(len(row) for row in data.rows)
  for row in data.rows: assert type(first(row)) is int
  [print(col) for col in data.cols.all]
  nums = adds(ydist(row,data) for row in data.rows)
  print(o(mu=nums.mu, sd=nums.sd))

def eg__ydist(file):
  data=Data(csv(file or the.file))
  r = data.rows[1] # ydist(data.rows[1],data))
  print(show(r),  ydist(r,data),the.p)
  #print(sorted(round(ydist(row,data),2) for row in data.rows))

def dump(d): print(len(d.rows)); [print(col) for col in d.cols.all]

def eg__addSub(file):
  data=Data(csv(file or the.file))
  dump(data)
  cached=data.rows[:]
  while data.rows: sub(data.rows.pop(), data)
  dump(data)
  for row in cached: add(row,data)
  dump(data)
  for row in data.rows: assert -17 < like(row,data,1000,2) < -10

def eg__clone(file):
  data=Data(csv(file or the.file))
  dump(data)
  data2=clone(data,src=data.rows)
  dump(data2)

def eg__stats(_):
   def c(b): return 1 if b else 0
   G  = random.gauss
   R  = random.random
   n  = 50
   b4 = [G(10,1) for _ in range(n)]
   d  = 0
   while d < 2:
     now = [x+d*R() for x in b4]
     b1  = cliffs(b4,now)
     b2  = bootstrap(b4,now)
     showd(o(d=d,cliffs=c(b1), boot=c(b2), agree=c(b1==b2)))
     d  += 0.1

def eg__rank(_):
   G  = random.gauss
   n=100
   d=dict(asIs  = [G(10,1) for _ in range(n)],
          copy1 = [G(20,1) for _ in range(n)],
          now1  = [G(20,1) for _ in range(n)],
          copy2 = [G(40,1) for _ in range(n)],
          now2  = [G(40,1) for _ in range(n)])
   for k,num in vals2RankedNums(d,the.Cohen).items():
      showd(o(what=num.txt, rank=num.rank, num=num.mu))

def eg__actLearn(file,  repeats=30):
  file = file or the.file
  name = re.search(r'([^/]+)\.csv$', file).group(1)
  data = Data(csv(file))
  b4   = yNums(data.rows,data)
  now  = Num()
  t1   = time.perf_counter_ns()
  for _ in range(repeats):
    add(ydist(first(actLearn(data, shuffle=True).best.rows ) ,data), now)
  t2  = time.perf_counter_ns()
  print(o(win= (b4.mu - now.mu) /(b4.mu - b4.lo),
          rows=len(data.rows),x=len(data.cols.x),y=len(data.cols.y),
          lo0=b4.lo, mu0=b4.mu, hi0=b4.hi, mu1=now.mu,sd1=now.sd,
          ms = int((t2-t1)/repeats/10**6),
          stop=the.Stop,name=name))

def eg__fast(file):
  def rx1(data):
    return ydist( first(actLearn(data,shuffle=True).best.rows), data)
  experiment1(file or the.file,
              repeats=30, 
              samples=[64,32,16,8],
              fun=rx1)

def eg__quick(file):
  def rx1(data):
    return [ydist(first(actLearn(data, shuffle=True).best.rows), data)]
  experiment1(file or the.file,
              repeats=10, 
              samples=[40,20,16,8],
              fun=rx1)

def eg__acts(file):
  def rx1(data):
    return [ydist(first(actLearn(data, shuffle=True).best.rows), data)]
  experiment1(file or the.file,
              repeats=30, 
              samples=[200,100,50,40,30,20,16,8],
              fun=rx1)

def experiment1(file, 
                repeats=30, samples=[32,16,8],
                fun=lambda d: ydist(first(actLearn(d,shuffle=True).best.rows),d)):
  name = re.search(r'([^/]+)\.csv$', file).group(1)
  data = Data(csv(file))
  rx   = dict(b4 = [ydist(row,data) for row in data.rows])
  asIs = adds(rx["b4"])
  t1   = time.perf_counter_ns()
  for the.Stop in samples:
    rx[the.Stop] = []
    for _ in range(repeats): rx[the.Stop] +=  fun(data) 
  t2 = time.perf_counter_ns()
  report = dict(rows = len(data.rows), 
                lo   = f"{asIs.lo:.2f}",
                x    = len(data.cols.x), 
                y    = len(data.cols.y),
                ms   = round((t2 - t1) / (repeats * len(samples) * 10**6)))
  order = vals2RankedNums(rx, asIs.sd*the.Cohen)
  for k in rx:
    v = order[k]
    report[k] = f"{v.rank} {v.mu:.2f} "
  report["name"]=name
  print("#"+str(list(report.keys())))
  print(list(report.values()))

def fname(f): return re.sub(".*/", "", f)

def eg__tree(file):
  data = Data(csv(file or the.file))
  model  = actLearn(data)
  b4  = yNums(data.rows,data)
  now = yNums(model.best.rows,data)
  nodes = tree(model.best.rows + model.rest.rows,data)
  print("\n"+fname(file or the.file))
  showd(o(mu1=b4.mu, mu2=now.mu,  sd1=b4.sd, sd2=now.sd))
  showTree(nodes)

def eg__rules(file):
  data  = Data(csv(file or the.file))
  b4    = yNums(data.rows, data)
  model = actLearn(data)
  now   = yNums(model.best.rows, data)
  nodes = tree(model.best.rows + model.rest.rows,data)
  todo  = yNums(model.todo, data)
  guess = sorted([(leaf(nodes,row).ys,row) for row in model.todo],key=first)
  mid = len(guess)//5
  after = yNums([row2 for row1 in model.todo for row2 in leaf(nodes,row1).rows],data)
  print(fname(file or the.file))
  print(o(txt1="b4", txt2="now",  txt3="todo",  txt4="after"))
  print(o(mu1=b4.mu, mu2=now.mu,  mu3=todo.mu,  mu4=ydist(guess[mid][1],data)))
  print(o(lo1=b4.lo, lo2=now.lo,  lo3=todo.lo,  lo4=ydist(guess[0][1],data)))
  print(o(hi1=b4.hi, hi2=now.hi,  hi3=todo.hi,  hi4=ydist(guess[-1][1],data)))
  print(o(n1=b4.n,   n2=now.n,    n3=todo.n,    n4=after.n))

def eg__afterDumb(file) : eg__after(file,repeats=30, smart=False)

def eg__after(file,repeats=30, smart=True):
  data  = Data(csv(file or the.file))
  b4    = yNums(data.rows, data) 
  overall= {j:Num() for j in [256,128,64,32,16,8]}
  for Stop in overall:
    the.Stop = Stop
    after = {j:Num() for j in [20,15,10,5,3,1]}
    learnt = Num()
    rand =Num()
    for _ in range(repeats):
      model = actLearn(data,shuffle=True)
      nodes = tree(model.best.rows + model.rest.rows,data)
      add(ydist(model.best.rows[0],data), learnt)
      guesses = sorted([(leaf(nodes,row).ys,row) for row in model.todo],key=first)
      for k in after:
        if smart:
              smart = min([(ydist(guess,data),guess) for _,guess in guesses[:k]], 
                           key=first)[1]
              add(ydist(smart,data),after[k]) 
        else:
              dumb = min([(ydist(row,data),row) for row in random.choices(model.todo,k=k)],
                   key=first)[1]
              add(ydist(dumb,data),after[k]) 
    def win(x): return str(round(100*(1 - (x - b4.lo)/(b4.mu - b4.lo))))
    print(the.Stop, win(learnt.mu), 
          " ".join([win(after[k].mu) for k in after]), 
          1, 
          fname(file or the.file), "smart" if smart else "dumb")

def eg__nbfew(file):
    data = Data(csv(file or the.file))
    print(f"runnning in {the.Mode} mode on file {file}")
    class_col_idx = data.cols.klass.at
    is_numeric = isinstance(data.rows[0][class_col_idx], (int, float))
    positive_value = 1 if is_numeric else "yes"
    positive_samples = [row for row in data.rows if row[class_col_idx] == positive_value]
    negative_samples = [row for row in data.rows if row[class_col_idx] != positive_value]
    all_n_pos_values = [32]
    n_pos_values = [n for n in all_n_pos_values if n <= len(positive_samples)]
    if not n_pos_values:
        return
    results = []
    for n_pos in n_pos_values:
        recalls = []
        precisions = []
        for i in range(100):
            selected_pos = random.sample(positive_samples, n_pos)
            n_neg = n_pos * 4
            if len(negative_samples) < n_neg:
                return
            selected_neg = random.sample(negative_samples, n_neg)
            pos_dataset = clone(data, selected_pos)
            neg_dataset = clone(data, selected_neg)
            test_samples = [row for row in data.rows if row not in selected_pos + selected_neg]
            tp = fp = fn = tn = 0
            for row in test_samples:
                actual_class = row[class_col_idx]
                best_dataset = likes(row, [pos_dataset, neg_dataset])
                predicted_class = (
                    positive_value if best_dataset is pos_dataset else (0 if is_numeric else "no")
                )
                if actual_class == positive_value and predicted_class == positive_value:
                    tp += 1
                elif actual_class == positive_value and predicted_class != positive_value:
                    fn += 1
                elif actual_class != positive_value and predicted_class == positive_value:
                    fp += 1
                else:
                    tn += 1
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            recalls.append(recall)
            precisions.append(precision)
        recalls = [r for r in recalls if r is not None]
        precisions = [p for p in precisions if p is not None]
        if not recalls:
            continue
        sorted_recalls = sorted(recalls)
        sorted_precisions = sorted(precisions)
        recall_median = sorted_recalls[len(sorted_recalls) // 2]
        recall_q1 = sorted_recalls[len(sorted_recalls) // 4]
        recall_q3 = sorted_recalls[3 * len(sorted_recalls) // 4]
        precision_median = sorted_precisions[len(sorted_precisions) // 2] if sorted_precisions else None
        precision_q1 = sorted_precisions[len(sorted_precisions) // 4] if sorted_precisions else None
        precision_q3 = sorted_precisions[3 * len(sorted_precisions) // 4] if sorted_precisions else None
        results.append([
            f"{recall_q1:.3f}", f"{recall_median:.3f}", f"{recall_q3:.3f}",
            f"{precision_q1:.3f}", f"{precision_median:.3f}", f"{precision_q3:.3f}"
        ])
    results_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(results_dir, exist_ok=True)
    csv_filename = os.path.join(results_dir, f"results_{os.path.splitext(os.path.basename(file))[0]}_{str(the.Mode)}.csv")
    with open(csv_filename, "w") as f:
        f.write("recall_Q1,recall_median,recall_Q3,precision_Q1,precision_median,precision_Q3\n")
        for row in results:
            f.write(",".join(row) + "\n")

def active_learning_uncertainty_loop(data, n_pos=8, repeats=10):
    class_col_idx = data.cols.klass.at
    class_values = list(set(row[class_col_idx] for row in data.rows))
    assert len(class_values) == 2, "This function assumes exactly 2 classes."
    # Determine which class is positive (minority) and which is negative (majority)
    class_counts = {}
    for val in class_values:
        class_counts[val] = sum(1 for row in data.rows if row[class_col_idx] == val)
    # Minority class is positive, majority class is negative
    pos = min(class_counts, key=class_counts.get)
    neg = max(class_counts, key=class_counts.get)
    results = []
    initial_q = 1
    final_q = 0
    batch_size = 1
    for i in range(repeats):
        print(f"Running eg__nbAL with n_pos={n_pos} and repeats={repeats} for iteration {i}")
        positive_samples = [row for row in data.rows if row[class_col_idx] == pos]
        negative_samples = [row for row in data.rows if row[class_col_idx] == neg]
        print(f"Found {len(positive_samples)} positive and {len(negative_samples)} negative samples")
        if len(positive_samples) < n_pos or len(negative_samples) < n_pos * 4:
            print(f"Skipping iteration {i}: insufficient samples (pos: {len(positive_samples)}, neg: {len(negative_samples)})")
            continue
        selected_pos = random.sample(positive_samples, n_pos)
        selected_neg = random.sample(negative_samples, n_pos * 4)
        labeled = selected_pos + selected_neg
        pool = [row for row in data.rows if row not in labeled]
        step_metrics = []
        no_iterations = len(pool)
        acq = 0
        # Initial evaluation at step 0
        datasets = []
        for val in class_values:
            datasets.append(clone(data, [row for row in labeled if row[class_col_idx] == val]))
        # if the.Mode == 'sklearn':
            # calculate_global_category_stats(datasets) # TODO: add this back in when Sym is implemented correctly
        tp = fp = fn = tn = 0
        print(f"Evaluating at step {acq} on all rows")
        for row in data.rows:  # <-- changed from 'for row in pool:'
            predicted_dataset = likes(row, datasets)
            predicted_class = predicted_dataset.rows[0][class_col_idx] if predicted_dataset.rows else None
            actual_class = row[class_col_idx]
            if predicted_class == pos and actual_class == pos:
                tp += 1
            elif predicted_class == pos and actual_class == neg:
                fp += 1
            elif predicted_class == neg and actual_class == pos:
                fn += 1
            elif predicted_class == neg and actual_class == neg:
                tn += 1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        false_alarm_pct = (fp / (fp + tn) * 100) if (fp + tn) > 0 else 0
        print(f"Repeat {i} Step {acq}: Precision: {precision:.4f}, Recall: {recall:.4f}, False alarm %: {false_alarm_pct:.2f}%")
        step_metrics.append((precision, recall, false_alarm_pct))
        while pool:
            # Acquire batch_size samples at once (trading accuracy for speed)
            batch_to_acquire = min(batch_size, len(pool))
            acquired_samples = []
            # Compute acquisition scores for all remaining pool samples
            q = initial_q - (initial_q - final_q) * acq / (no_iterations if no_iterations > 0 else 1)
            pool_scores = []
            for row in pool:
                logps = []
                for dset in datasets:
                    logp = like(row, dset, nall=sum(d.n for d in datasets), nh=len(datasets))
                    logps.append(logp)
                max_logp = max(logps)
                probs = [math.exp(lp - max_logp) for lp in logps]
                total = sum(probs)
                probs = [p / total for p in probs]
                best = max(probs)
                rest = min(probs)
                numerator = best + q * rest
                denominator = abs(q * best - rest) if abs(q * best - rest) > 1e-12 else 1e-12
                pool_scores.append(numerator / denominator)
            # Select top batch_size samples at once
            for _ in range(batch_to_acquire):
                if not pool:
                    break
                # Find the sample with highest score
                best_idx = max(range(len(pool)), key=lambda i: pool_scores[i])
                acquired_samples.append(pool[best_idx])
                # Remove from pool and scores
                pool.pop(best_idx)
                pool_scores.pop(best_idx)
                acq += 1
            # Add acquired samples to labeled set
            labeled.extend(acquired_samples)
            # Update datasets incrementally (add new samples to existing datasets)
            for val_idx, val in enumerate(class_values):
                new_samples = [row for row in acquired_samples if row[class_col_idx] == val]
                for row in new_samples:
                    add(row, datasets[val_idx])
            # if the.Mode == 'sklearn':
            #     calculate_global_category_stats(datasets)
            # Evaluate after each batch (except the first one which was already evaluated)
            if acq % batch_size == 0 and acq > 0:
                tp = fp = fn = tn = 0
                print(f"Evaluating at step {acq} on all rows")
                for row in data.rows:  # <-- changed from 'for row in pool:'
                    predicted_dataset = likes(row, datasets)
                    predicted_class = predicted_dataset.rows[0][class_col_idx] if predicted_dataset.rows else None
                    actual_class = row[class_col_idx]
                    if predicted_class == pos and actual_class == pos:
                        tp += 1
                    elif predicted_class == pos and actual_class == neg:
                        fp += 1
                    elif predicted_class == neg and actual_class == pos:
                        fn += 1
                    elif predicted_class == neg and actual_class == neg:
                        tn += 1
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                false_alarm_pct = (fp / (fp + tn) * 100) if (fp + tn) > 0 else 0
                print(f"Repeat {i} Step {acq}: Precision: {precision:.4f}, Recall: {recall:.4f}, False alarm %: {false_alarm_pct:.2f}%")
                step_metrics.append((precision, recall, false_alarm_pct))
            elif acq > 0:
                # Set precision and recall to 0 for non-evaluation steps
                step_metrics.append((0, 0, 0))
            if not pool:
                break
        # Final evaluation after all samples have been acquired
        tp = fp = fn = 0
        tn = 0
        print(f"Final evaluation with all samples acquired on all rows")
        for row in data.rows:
            predicted_dataset = likes(row, datasets)
            predicted_class = predicted_dataset.rows[0][class_col_idx] if predicted_dataset.rows else None
            actual_class = row[class_col_idx]
            if predicted_class == pos and actual_class == pos:
                tp += 1
            elif predicted_class == pos and actual_class == neg:
                fp += 1
            elif predicted_class == neg and actual_class == pos:
                fn += 1
            elif predicted_class == neg and actual_class == neg:
                tn += 1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        false_alarm_pct = (fp / (fp + tn) * 100) if (fp + tn) > 0 else 0
        print(f"Final Step: Precision: {precision:.4f}, Recall: {recall:.4f}, False alarm %: {false_alarm_pct:.2f}%")
        step_metrics.append((precision, recall, false_alarm_pct))
        results.append(step_metrics)
    return results

# Update eg__nbAL to optionally use the uncertainty loop
# Usage: eg__nbAL(file, repeats=100, acq_mode='uncertainty')
def eg__nbAL(file, repeats=20):
    n_pos_val = the.n_pos
    data = Data(csv(file or the.file))
    base_filename = os.path.splitext(os.path.basename(file))[0]
    # n_pos_values = [8, 16, 32]
    # for n_pos_val in n_pos_values:
    print(f"Running eg__nbAL on {file} with n_pos={n_pos_val} and repeats={repeats}")
    results = active_learning_uncertainty_loop(data, n_pos=n_pos_val, repeats=repeats)
    if results:
        # Compute correct number of steps: (total samples - initial labeled set size) + 2
        N_total = len(data.rows)
        n_neg = n_pos_val * 4
        N_labeled = n_pos_val + n_neg
        n_steps = (N_total - N_labeled) + 2
        # Pad runs to n_steps if needed
        for run in results:
            while len(run) < n_steps:
                run.append((None, None, None))
        # For each step, collect all recalls, precisions, and false alarm rates across repeats
        step_recalls = [[] for _ in range(n_steps)]
        step_precisions = [[] for _ in range(n_steps)]
        step_false_alarms = [[] for _ in range(n_steps)]
        for run in results:
            for step in range(n_steps):
                if run[step][0] is not None:
                    step_precisions[step].append(run[step][0])
                    step_recalls[step].append(run[step][1])
                    step_false_alarms[step].append(run[step][2])
        # Compute Q1, median, Q3 for each step
        def get_quartiles(lst):
            lst_sorted = sorted(lst)
            n = len(lst_sorted)
            q1 = lst_sorted[n // 4] if n > 0 else 0
            median = lst_sorted[n // 2] if n > 0 else 0
            q3 = lst_sorted[(3 * n) // 4] if n > 0 else 0
            return q1, median, q3
        # Save to CSV
        results_dir = os.path.join(os.getcwd(), "results_al")
        os.makedirs(results_dir, exist_ok=True)
        csv_filename = os.path.join(
            results_dir,
            f"results_uncertainty_{str(the.Mode)}_{n_pos_val}_{base_filename}.csv"
        )
        with open(csv_filename, "w") as f:
            f.write("step,recall_Q1,recall_median,recall_Q3,precision_Q1,precision_median,precision_Q3,false_alarm_Q1,false_alarm_median,false_alarm_Q3\n")
            for step in range(n_steps):
                recall_q1, recall_median, recall_q3 = get_quartiles(step_recalls[step])
                precision_q1, precision_median, precision_q3 = get_quartiles(step_precisions[step])
                false_alarm_q1, false_alarm_median, false_alarm_q3 = get_quartiles(step_false_alarms[step])
                f.write(f"{step},{recall_q1:.4f},{recall_median:.4f},{recall_q3:.4f},{precision_q1:.4f},{precision_median:.4f},{precision_q3:.4f},{false_alarm_q1:.4f},{false_alarm_median:.4f},{false_alarm_q3:.4f}\n")
    return

#--------- --------- --------- --------- --------- --------- ------- -------
regx = r"-\w+\s*(\w+).*=\s*(\S+)"
the  = o(**{m[1]:coerce(m[2]) for m in re.finditer(regx,__doc__)})
random.seed(the.rseed)

def eg__timecheck(file):
    """
    Loads the given CSV, trains on the entire dataset, then performs inference on the entire dataset.
    For 10 repeats:
      - Train on all data (build datasets by class)
      - Perform inference on all data
      - Measure and report average inference (prediction) time only
    At the end, print the average inference time (ms) for each mode.
    """
    data = Data(csv(file or the.file))
    n_repeats = 10
    class_col_idx = data.cols.klass.at
    all_rows = data.rows[:]
    for mode in ['bl', 'sklearn']:
        the.Mode = mode
        predict_times = []
        for i in range(n_repeats):
            print(f"Running eg__timecheck for mode {mode} for repeat {i}")
            # Train on entire dataset
            class_vals = list(set(row[class_col_idx] for row in all_rows))
            datasets = [clone(data, [row for row in all_rows if row[class_col_idx] == v]) for v in class_vals]
            if the.Mode == 'sklearn':
                calculate_global_category_stats(datasets)
            
            # Measure inference time on entire dataset
            t0 = time.perf_counter()
            # --- Begin evaluation for false alarm calculation ---
            tp = fp = fn = tn = 0
            for row in all_rows:
                predicted_dataset = likes(row, datasets)
                predicted_class = predicted_dataset.rows[0][class_col_idx] if predicted_dataset.rows else None
                actual_class = row[class_col_idx]
                # Assume the first class in class_vals is positive, second is negative
                pos = class_vals[0]
                neg = class_vals[1] if len(class_vals) > 1 else None
                if actual_class == pos and predicted_class == pos:
                    tp += 1
                elif actual_class == pos and predicted_class == neg:
                    fn += 1
                elif actual_class == neg and predicted_class == pos:
                    fp += 1
                elif actual_class == neg and predicted_class == neg:
                    tn += 1
            t1 = time.perf_counter()
            predict_times.append((t1 - t0) * 1000)  # ms
            # Calculate false alarm %
            false_alarm_pct = (fp / (fp + tn) * 100) if (fp + tn) > 0 else 0
            print(f"Predict time for repeat {i}: {predict_times[-1]} ms, False alarm %: {false_alarm_pct:.2f}%")
        mean_pred = sum(predict_times) / len(predict_times)
        print(o(mode=mode, avg_inference_time_ms=mean_pred, n=n_repeats))

def eg__naive(file):
    """
    Trains on all samples and tests on all samples. Prints a table with tn, fn, fp, tp, pd, prec, pf, acc for each label and overall.
    """
    data = Data(csv(file or the.file))
    class_col_idx = data.cols.klass.at
    class_values = list(sorted(set(row[class_col_idx] for row in data.rows)))
    datasets = [clone(data, [row for row in data.rows if row[class_col_idx] == v]) for v in class_values]
    # if the.Mode == 'sklearn':
    #     calculate_global_category_stats(datasets)
    # Prepare confusion matrix for each class
    results = {v: {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0} for v in class_values}
    for row in data.rows:
        actual = row[class_col_idx]
        predicted_dataset = likes(row, datasets)
        predicted = predicted_dataset.rows[0][class_col_idx] if predicted_dataset.rows else None
        for v in class_values:
            if actual == v and predicted == v:
                results[v]['tp'] += 1
            elif actual == v and predicted != v:
                results[v]['fn'] += 1
            elif actual != v and predicted == v:
                results[v]['fp'] += 1
            elif actual != v and predicted != v:
                results[v]['tn'] += 1
    # Compute metrics and print table
    def safe_div(a, b):
        return a / b if b else 0
    print(f"{'Label':>10} {'TN':>6} {'FN':>6} {'FP':>6} {'TP':>6} {'PD':>8} {'Prec':>8} {'PF':>8} {'Acc':>8}")
    print(f"{'-'*10} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    total = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
    for v in class_values:
        tp = results[v]['tp']
        fp = results[v]['fp']
        tn = results[v]['tn']
        fn = results[v]['fn']
        total['tp'] += tp
        total['fp'] += fp
        total['tn'] += tn
        total['fn'] += fn
        pd = safe_div(tp, tp + fn)  # recall
        prec = safe_div(tp, tp + fp)
        pf = safe_div(fp, fp + tn)
        acc = safe_div(tp + tn, tp + tn + fp + fn)
        print(f"{str(v):>10} {tn:6} {fn:6} {fp:6} {tp:6} {pd:8.3f} {prec:8.3f} {pf:8.3f} {acc:8.3f}")
    # Overall
    tp = total['tp']
    fp = total['fp']
    tn = total['tn']
    fn = total['fn']
    pd = safe_div(tp, tp + fn)
    prec = safe_div(tp, tp + fp)
    pf = safe_div(fp, fp + tn)
    acc = safe_div(tp + tn, tp + tn + fp + fn)
    print(f"{'ALL':>10} {tn:6} {fn:6} {fp:6} {tp:6} {pd:8.3f} {prec:8.3f} {pf:8.3f} {acc:8.3f}")

if __name__ == "__main__":  main()