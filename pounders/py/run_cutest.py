# from cutest_adapter import load_cutest
# from full_demo_aug7 import run

# for nm in ['HS21', 'HS35MOD', 'HS12', 'HS35', 'HUBFIT', 'HS76', 'HS44', 'HS84',
#            'DIPIGRI', 'HS100', 'HS113']:
#     prob = load_cutest(nm)
#     print(f"\n{'='*60}\n{nm}: n={prob['n']} rows={prob['m_rows']}\n{'='*60}")
#     run(f=prob['f'], constraints=prob['constraints'], x0=prob['x0'],
#         low=prob['low'], upp=prob['upp'], delta0=1.0, delta_max=1.0,
#         verbose=False)   # quiet — just the final summary per problem




from .cutest_adapter import load_cutest
from .active_set_trsqp import run

results = {}
for nm in ['HS21', 'HS35MOD', 'HS12', 'HS35', 'HUBFIT', 'HS76', 'HS44', 'HS84',
           'DIPIGRI', 'HS100', 'HS113']:
    try:
        prob = load_cutest(nm)
    except Exception as e:
        print(f"\n{nm}: LOAD FAILED {type(e).__name__}: {e}")
        continue

    print(f"\n{'='*60}\n{nm}: n={prob['n']} rows={prob['m_rows']}\n{'='*60}")
    try:
        results[nm] = run(f=prob['f'], constraints=prob['constraints'],
                          x0=prob['x0'], low=prob['low'], upp=prob['upp'],
                          delta0=1.0, delta_max=1.0, verbose=False)
    except Exception as e:
        print(f"  !! FAILED: {type(e).__name__}: {e}")

print(f"\n{'prob':>9} {'growth':>7} {'outer':>6} {'evals':>7}  term")
for nm, r in results.items():
    print(f"{nm:>9} {r['growth']:>7} {r['outer']:>6} "
          f"{r['evf']+r['evc']:>7}  {r['term_reason'].split(':')[0]}")