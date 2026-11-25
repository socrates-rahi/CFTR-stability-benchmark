#!/usr/bin/env python3
"""
DDMut toolkit for the CFTR NBD benchmark (59 mutations).

Written from the DDMut API documentation (https://biosig.lab.uq.edu.au/ddmut/api).
The network parts (`run`) are UNTESTED against the live server: run `run --limit 3`
and compare with the web form before launching everything.

Subcommands
  make-inputs   write mutation lists + a manifest from bahia_reference_59.csv
  check-pdb     verify wild-type residue identity/numbering in a PDB for a list file
  run           submit a list file for one structure, poll, save raw JSON (resumable)
  parse         flatten saved JSON into a CSV of ddmut_fwd / ddmut_rev

Typical order
  python3 ddmut_toolkit.py make-inputs bahia_reference_59.csv --chain A
  python3 ddmut_toolkit.py check-pdb 2PZE_prepared.pdb ddmut_inputs/nbd1_list.txt
  python3 ddmut_toolkit.py run 2PZE_prepared.pdb ddmut_inputs/nbd1_list.txt --tag nbd1 --limit 3
  python3 ddmut_toolkit.py run 2PZE_prepared.pdb ddmut_inputs/nbd1_list.txt --tag nbd1
  (NBD2: one PDB per construct -> run once per ddmut_inputs/nbd2_construct_N_list.txt)
  python3 ddmut_toolkit.py parse ddmut_raw benchmark_in.csv benchmark.csv
"""
import argparse, csv, json, os, re, sys, time

AA3 = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
       'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V',
       'HID':'H','HIE':'H','HIP':'H','CYX':'C'}
MUT = re.compile(r'^([A-Z])(\d+)([A-Z])$')

def read_rows(path):
    return list(csv.DictReader(open(path)))

# ---------------------------------------------------------------- make-inputs
def make_inputs(a):
    rows = read_rows(a.csv)
    os.makedirs(a.out, exist_ok=True)
    manifest, groups = [], {}
    for r in rows:
        if not MUT.match(r['mutation']): sys.exit('bad mutation ' + r['mutation'])
        key = 'nbd1' if r['domain'] == 'NBD1' else f"nbd2_construct_{r['construct']}"
        groups.setdefault(key, []).append(r)
    for key, rs in groups.items():
        with open(f'{a.out}/{key}_list.txt', 'w') as f:
            for r in rs: f.write(f"{a.chain} {r['mutation']}\n")   # "chain wt pos mut" per DDMut help page
        for r in rs:
            m = MUT.match(r['mutation'])
            manifest.append(dict(id=r['id'], domain=r['domain'], construct=r.get('construct', ''), mutation=r['mutation'],
                                 required_wt_in_structure=m.group(1), position=m.group(2), list_file=f'{key}_list.txt'))
    with open(f'{a.out}/manifest.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0])); w.writeheader(); w.writerows(manifest)
    print('wrote', len(groups), 'list files:', ', '.join(f'{k}({len(v)})' for k, v in groups.items()))
    print('NBD2 note: each construct list needs a PDB carrying THAT background (see Bahia Table 3);')
    print('construct_11 is identical to 9sol = 6UK1 as deposited/prepared.')

# ---------------------------------------------------------------- check-pdb
def pdb_residues(path):
    res = {}
    for line in open(path):
        if line.startswith(('ATOM', 'HETATM')) and line[12:16].strip() == 'CA':
            res[(line[21], int(line[22:26]))] = AA3.get(line[17:20].strip(), '?')
    return res

def check_pdb(a):
    res = pdb_residues(a.pdb); bad = 0; n = 0
    chains = sorted({c for c, _ in res})
    print('chains with CA atoms:', chains)
    for line in open(a.list):
        if not line.strip(): continue
        chain, mut = line.split(); m = MUT.match(mut); n += 1
        found = res.get((chain, int(m.group(2))))
        if found != m.group(1):
            bad += 1
            print(f'  MISMATCH {chain} {mut}: structure has {found!r} at {m.group(2)}, expected {m.group(1)!r}')
    print(f'{n - bad}/{n} entries match the structure' + ('  <-- fix numbering/chain/construct before running' if bad else '  OK'))
    sys.exit(1 if bad else 0)

# ---------------------------------------------------------------- run
def run(a):
    import requests
    base = 'https://biosig.lab.uq.edu.au/ddmut/api/'
    os.makedirs(a.raw, exist_ok=True)
    lines = [l.strip() for l in open(a.list) if l.strip()]
    if a.limit: lines = lines[:a.limit]
    tmp = f'{a.raw}/{a.tag}_submitted_list.txt'
    open(tmp, 'w').write('\n'.join(lines) + '\n')
    out = f'{a.raw}/{a.tag}.json'
    if os.path.exists(out) and not a.force:
        sys.exit(f'{out} exists (use --force to overwrite); resumable by design: raw files are never lost')
    with open(a.pdb, 'rb') as pf, open(tmp, 'rb') as mf:
        r = requests.post(base + 'prediction_list',
                          files={'pdb_file': pf, 'mutation_list': mf},
                          data={'reverse': 'True' if a.reverse else 'False'}, timeout=120)
    r.raise_for_status()
    job = r.json().get('job_id'); print('job_id', job)
    open(f'{a.raw}/{a.tag}_job.txt', 'w').write(str(job))
    for _ in range(a.max_polls):
        time.sleep(a.poll)
        g = requests.get(base + 'prediction_list', data={'job_id': job}, timeout=120)  # docs show GET with job_id form field
        js = g.json()
        if isinstance(js, dict) and js.get('message') == 'RUNNING':
            print('...running'); continue
        json.dump(js, open(out, 'w'), indent=1); print('saved', out); return
    sys.exit('timed out waiting; job id saved, re-poll later')

# ---------------------------------------------------------------- parse
def find_numbers(obj):
    """tolerant: yield (mutation_key, forward, reverse) from whatever JSON shape the list endpoint returns."""
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, dict) and ('prediction' in v):
                out.append((k, v.get('prediction'), v.get('prediction_reverse')))
            else:
                out += find_numbers(v)
    elif isinstance(obj, list):
        for v in obj:
            if isinstance(v, dict) and 'prediction' in v:
                key = ''.join(str(v.get(x, '')) for x in ('chain', 'wild-type', 'res_number', 'mutant'))
                out.append((key, v.get('prediction'), v.get('prediction_reverse')))
            else:
                out += find_numbers(v)
    return out

def parse(a):
    rows = read_rows(a.base_csv)
    preds = {}
    for fn in sorted(os.listdir(a.raw)):
        if not fn.endswith('.json'): continue
        tag = fn[:-5]
        for key, fwd, rev in find_numbers(json.load(open(f'{a.raw}/{fn}'))):
            m = re.search(r'([A-Z])(\d+)([A-Z])\s*$', str(key).replace(' ', '')) or re.search(r'([A-Z])(\d+)([A-Z])', str(key))
            if m: preds[(tag, m.group(0))] = (fwd, rev)
    miss = 0
    for r in rows:
        tag = 'nbd1' if r['domain'] == 'NBD1' else f"nbd2_construct_{r['construct']}"
        v = preds.get((tag, r['mutation']))
        if v is None: miss += 1; r['ddmut_fwd'] = ''; r['ddmut_rev'] = ''
        else: r['ddmut_fwd'], r['ddmut_rev'] = v
    with open(a.out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f'wrote {a.out_csv}: {len(rows) - miss}/{len(rows)} rows have DDMut predictions')

# ---------------------------------------------------------------- cli
p = argparse.ArgumentParser(); s = p.add_subparsers(dest='cmd', required=True)
x = s.add_parser('make-inputs'); x.add_argument('csv'); x.add_argument('--chain', default='A'); x.add_argument('--out', default='ddmut_inputs'); x.set_defaults(f=make_inputs)
x = s.add_parser('check-pdb'); x.add_argument('pdb'); x.add_argument('list'); x.set_defaults(f=check_pdb)
x = s.add_parser('run'); x.add_argument('pdb'); x.add_argument('list'); x.add_argument('--tag', required=True)
x.add_argument('--raw', default='ddmut_raw'); x.add_argument('--limit', type=int, default=0); x.add_argument('--reverse', action='store_true')
x.add_argument('--poll', type=int, default=20); x.add_argument('--max-polls', type=int, default=180); x.add_argument('--force', action='store_true'); x.set_defaults(f=run)
x = s.add_parser('parse'); x.add_argument('raw'); x.add_argument('base_csv'); x.add_argument('out_csv'); x.set_defaults(f=parse)
a = p.parse_args(); a.f(a)
