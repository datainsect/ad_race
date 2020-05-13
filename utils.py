def myeval(s):
    l = eval(s)
    if len(l)==0:
        l.append(1)
    return l
