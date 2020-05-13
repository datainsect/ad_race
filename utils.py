


def myeval(s):
    l = eval(s)
    if len(l)==0:
        l.append(1)
    return l


def rmnan(x):
    if not x or len(x)==0:
        return x
    return x.replace(', nan','').replace('nan,','').replace('nan','')