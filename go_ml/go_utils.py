from goatools.obo_parser import GODag
from goatools.godag.go_tasks import get_go2parents

godag = GODag('../go-basic.obo')
go2parents_isa = get_go2parents(godag, set())
def get_ancestors(go, go2parents):
    seen = set()
    b = {go}
    while b:
        next_term = b.pop()
        if(next_term in seen or not next_term in go2parents):
            continue
        seen.add(next_term)
        b.update(go2parents[next_term])
    return seen