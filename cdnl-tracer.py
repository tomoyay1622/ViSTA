import json
import sys
import os
from dataclasses import dataclass, field, asdict
from typing import Set, Dict, Optional, Tuple, List, cast, Union, Iterable
import itertools

# ==========================================
# 0. Tracing Infrastructure
# ==========================================

class SolverTracer:
    def __init__(self):
        self.events = []
        self.symbol_table = {}
        self.stats = {}  # for statistics

    def set_symbol_table(self, table: Dict[int, str]):
        self.symbol_table = table

    def set_stats(self, stats: Dict):
        self.stats = stats

    def log(self, event_type: str, details: Dict):
        self.events.append({
            "type": event_type,
            "details": details
        })

    def dump_json(self):
        data = {
            "symbols": self.symbol_table,
            "stats": self.stats,
            "events": self.events
        }
        print(json.dumps(data, indent=2))

# ==========================================
# 1. Core Data Structures
# ==========================================

@dataclass(frozen=True)
class Literal:
    id: int

    @property
    def atom_id(self) -> int:
        return abs(self.id)

    @property
    def is_positive(self) -> bool:
        return self.id > 0

    @property
    def complement(self) -> 'Literal':
        return Literal(-self.id)

    def __repr__(self) -> str:
        return f"{self.id}"
    
    def to_json(self):
        return self.id

    def __lt__(self, other):
        return self.id < other.id

Nogood = frozenset[Literal]

# ==========================================
# 2. Logic Program
# ==========================================

@dataclass
class Rule:
    head: int            # Head atom ID (0 if constraint)
    body_id: int         # Assigned Body ID (only for head > 0 rules)

class Program:
    def __init__(self):
        self.rules: List[Rule] = []
        self.atoms: Set[int] = set()
        self.symbol_table: Dict[int, str] = {}
        
        # Body Management
        self.body_to_id: Dict[frozenset[int], int] = {}
        self.id_to_body: Dict[int, List[int]] = {}
        self.max_atom_id: int = 0
        self.next_body_id: int = 0
        self.head_to_bodies: Dict[int, List[int]] = {}
        
        # Nogoods generated directly from integrity constraints (:- body.)
        self.constraint_nogoods: Set[Nogood] = set()

        # SCC related
        self.scc_map: Dict[int, int] = {}
        self.cyclic_atoms: Set[int] = set()

    def register_body(self, body_lits: List[int]) -> int:
        key = frozenset(body_lits)
        if key in self.body_to_id:
            return self.body_to_id[key]
        bid = self.next_body_id
        self.next_body_id += 1
        self.body_to_id[key] = bid
        self.id_to_body[bid] = body_lits
        for l in body_lits: self.atoms.add(abs(l))
        return bid

    def finalize_bodies(self):
        self.next_body_id = self.max_atom_id + 1

    def get_completion_nogoods(self) -> Set[Nogood]:
        """Generate Delta_Pi (Clark Completion) for standard rules only."""
        nogoods = set()
        
        # Body-oriented (Only for bodies that defined an atom)
        for bid, lits in self.id_to_body.items():
            # delta(beta) = {F(beta), T(l1), ..., T(ln)}
            d_beta = {Literal(-bid)}
            for l in lits: d_beta.add(Literal(l))
            nogoods.add(frozenset(d_beta))
            
            # Delta(beta) = { {T(beta), F(l1)}, ..., {T(beta), F(ln)} }
            for l in lits:
                ng = frozenset({Literal(bid), Literal(l).complement})
                nogoods.add(ng)

        # Atom-oriented
        for atom in self.atoms:
            bodies = self.head_to_bodies.get(atom, [])
            for bid in bodies:
                ng = frozenset({Literal(-atom), Literal(bid)})
                nogoods.add(ng)
            d_p = {Literal(atom)}
            for bid in bodies: d_p.add(Literal(-bid))
            nogoods.add(frozenset(d_p))
            
        return nogoods

    def compute_sccs(self):
        adj: Dict[int, Set[int]] = {a: set() for a in self.atoms}
        
        # Only use rules with actual heads (head > 0)
        for rule in self.rules:
            
            body_lits = self.id_to_body[rule.body_id]
            for lit in body_lits:
                if lit > 0 and lit in self.atoms:
                    adj[lit].add(rule.head)
        
        index = 0
        stack = []
        indices = {}
        lowlink = {}
        on_stack = set()
        scc_count = 0

        def strongconnect(v):
            nonlocal index, scc_count
            indices[v] = lowlink[v] = index
            index += 1
            stack.append(v)
            on_stack.add(v)
            
            for w in adj[v]:
                if w not in indices:
                    strongconnect(w)
                    lowlink[v] = min(lowlink[v], lowlink[w])
                elif w in on_stack:
                    lowlink[v] = min(lowlink[v], indices[w])
            if lowlink[v] == indices[v]:
                while True:
                    w = stack.pop()
                    on_stack.remove(w)
                    self.scc_map[w] = scc_count
                    if w == v: break
                scc_count += 1

        for v in self.atoms:
            if v not in indices: strongconnect(v)
        
        for atom in self.atoms:
            scc_id = self.scc_map[atom]
            is_cyc = False
            if atom in self.head_to_bodies:
                for bid in self.head_to_bodies[atom]:
                    body_lits = self.id_to_body[bid]
                    for lit in body_lits:
                        if lit > 0 and self.scc_map.get(lit) == scc_id:
                            is_cyc = True; break
                    if is_cyc: break
            if is_cyc: self.cyclic_atoms.add(atom)

def parse_aspif(aspif_text: str) -> Program:
    prog = Program()
    raw_rules = []
    
    for line in aspif_text.strip().splitlines():
        if not line or line.startswith("asp") or line.startswith("%"): continue
        parts = list(map(str, line.split()))
        if not parts: continue
        type_code = int(parts[0])
        
        if type_code == 1:
            idx = 1
            # head type, len, lits
            head_type = int(parts[idx]); idx += 1
            head_len = int(parts[idx]); idx += 1
            
            head_atom = 0
            if head_len > 0:
                head_atom = int(parts[idx]); idx += head_len
            if head_atom > prog.max_atom_id: prog.max_atom_id = head_atom
            
            # body type, len, lits
            body_type = int(parts[idx]); idx += 1
            body_len = int(parts[idx]); idx += 1
            body_lits = []
            for _ in range(body_len):
                lit = int(parts[idx]); idx += 1; body_lits.append(lit)
                if abs(lit) > prog.max_atom_id: prog.max_atom_id = abs(lit)
            raw_rules.append((head_atom, body_lits))
            
        elif type_code == 4:
            name = parts[2]
            atom_id = int(parts[-1])
            prog.symbol_table[atom_id] = name
            if atom_id > prog.max_atom_id: prog.max_atom_id = atom_id

    # 1. Initialize Body IDs
    prog.finalize_bodies()
    
    # 2. Process Rules, Separate Constraints
    for head, body_lits in raw_rules:
        if head == 0:
            # INTEGRITY CONSTRAINT: :- body.
            # Nogood is just the body itself: {T(l1), T(l2), ...}
            constraint_ng = frozenset(Literal(l) for l in body_lits)
            prog.constraint_nogoods.add(constraint_ng)
            
            # Add atoms in constraints to atoms list
            for l in body_lits:
                prog.atoms.add(abs(l))

        else:
            # REGULAR RULE: H :- body.
            bid = prog.register_body(body_lits)
            prog.rules.append(Rule(head=head, body_id=bid))
            
            prog.atoms.add(head)
            if head not in prog.head_to_bodies: prog.head_to_bodies[head] = []
            prog.head_to_bodies[head].append(bid)
    

    prog.compute_sccs()
    # Also register Body IDs in symbol table for visualization
    for bid, lits in prog.id_to_body.items():
        # Make a readable string for the body
        s = "{" + ",".join([str(l) for l in lits]) + "}"
        prog.symbol_table[bid] = s

    tracer.set_symbol_table(prog.symbol_table)
    return prog

# ==========================================
# 3. Solver Components
# ==========================================

class Assignment:
    def __init__(self, all_vars: Set[int]):
        self._assignments: Dict[int, Tuple[Literal, int, Optional[Nogood]]] = {}
        self._assignment_stack: List[Literal] = []
        self.literals: Set[Literal] = set()
        self.complements: Set[Literal] = set()
        self.all_vars = sorted(list(all_vars))

    def add(self, literal: Literal, level: int, antecedent: Optional[Nogood], reason_type: str = "propagation"):
        if self.is_assigned(literal.atom_id): return
        self._assignments[literal.atom_id] = (literal, level, antecedent)
        self._assignment_stack.append(literal)
        self.literals.add(literal)
        self.complements.add(literal.complement)
        
        # Log Assignment
        antecedent_ids = [l.id for l in antecedent] if antecedent else []
        tracer.log(reason_type, {
            "literal": literal.id,
            "level": level,
            "antecedent": antecedent_ids
        })

    def backjump(self, target_level: int):
        tracer.log("backjump", {"target_level": target_level})
        while self._assignment_stack:
            last = self._assignment_stack[-1]
            if self._assignments[last.atom_id][1] <= target_level: break
            lit = self._assignment_stack.pop()
            del self._assignments[lit.atom_id]
            self.literals.remove(lit)
            self.complements.remove(lit.complement)

    def is_assigned(self, atom_id: int) -> bool: return atom_id in self._assignments
    def is_false(self, atom_id: int) -> bool: return Literal(-atom_id) in self.literals
    def get_level(self, literal: Literal) -> int: return self._assignments[literal.atom_id][1] if literal.atom_id in self._assignments else -1
    def get_antecedent(self, literal: Literal) -> Optional[Nogood]: return self._assignments[literal.atom_id][2] if literal.atom_id in self._assignments else None
    
    def get_last_assigned_in_nogood(self, nogood: Nogood, dl: int) -> Optional[Literal]:
        for lit in reversed(self._assignment_stack):
            if lit in nogood and self.get_level(lit) == dl: return lit
        return None

    def is_total(self) -> bool: return len(self._assignments) == len(self.all_vars)
    def get_unassigned_var(self) -> Optional[int]:
        for v in self.all_vars:
            if v not in self._assignments: return v
        return None

class UnfoundedSetChecker:
    def __init__(self, program: Program):
        self.program = program
        self.source: Dict[int, Union[str, int]] = {} 
        for p in self.program.atoms:
            self.source[p] = "BOTTOM" if p in self.program.cyclic_atoms else "TOP"

    def get_external_bodies(self, U: Set[int]) -> Set[int]:
        eb = set()
        for p in U:
            if p in self.program.head_to_bodies:
                for bid in self.program.head_to_bodies[p]:
                    body_lits = self.program.id_to_body[bid]
                    pos_lits = {l for l in body_lits if l > 0}
                    if pos_lits.isdisjoint(U):
                        eb.add(bid)
        return eb

    def check(self, A: Assignment) -> Set[int]:
        tracer.log("us_check_start", {})
        S = set()
        for p in self.program.atoms:
            if A.is_false(p): continue
            src = self.source[p]
            invalid = False
            if src == "BOTTOM": invalid = True
            elif isinstance(src, int):
                if A.is_false(src): invalid = True
            if invalid: S.add(p)
            
        while True:
            T = set()
            for p in self.program.atoms:
                if A.is_false(p) or p in S: continue
                src = self.source[p]
                if isinstance(src, int):
                    body_lits = self.program.id_to_body[src]
                    scc_id = self.program.scc_map[p]
                    depends = False
                    for lit in body_lits:
                        if lit > 0 and lit in S and self.program.scc_map.get(lit) == scc_id:
                            depends = True; break
                    if depends: T.add(p)
            if not T: break
            S.update(T)

        if not S:
            tracer.log("no_us", {}) 
            return set()
        tracer.log("us_scope", {"scope": [(p, self.source[p]) for p in S]})
        
        while S:
            p = next(iter(S))
            U = {p}
            while True:
                eb = self.get_external_bodies(U)
                tracer.log("us_iterate", {"U": list(U), "EB": list(eb)})
                
                valid_beta = None
                all_false = True
                for bid in eb:
                    if not A.is_false(bid):
                        all_false = False; valid_beta = bid; break
                
                if all_false:
                    tracer.log("us_found", {"unfounded_set": list(U)})
                    return U
                
                beta_id = valid_beta
                beta_lits = self.program.id_to_body[beta_id]
                scc_id = self.program.scc_map[p]
                pos_lits = {l for l in beta_lits if l > 0}
                scc_p_S = {x for x in self.program.atoms if self.program.scc_map.get(x) == scc_id and x in S}
                C = pos_lits & scc_p_S
                
                if not C:
                    to_remove = set()
                    for q in U:
                        if q in self.program.head_to_bodies and beta_id in self.program.head_to_bodies[q]:
                             self.source[q] = beta_id
                             tracer.log("us_source_update", {"atom": q, "new_source": beta_id})
                             to_remove.add(q)
                    U.difference_update(to_remove)
                    S.difference_update(to_remove)
                else:
                    U.update(C)
                if not U: break
        tracer.log("no_us", {})
        return set()

# ==========================================
# 4. CDNL Algorithms
# ==========================================

def unit_propagation(dl: int, all_nogoods: Set[Nogood], A: Assignment) -> Tuple[str, Optional[Nogood]]:
    while True:
        unit_found = False
        for nogood in all_nogoods:
            if not nogood.isdisjoint(A.complements): continue 
            unassigned = nogood - A.literals
            if not unassigned:
                tracer.log("conflict", {"nogood": [l.id for l in nogood]})
                return "CONFLICT", nogood
            if len(unassigned) == 1:
                unit = next(iter(unassigned))
                to_assign = unit.complement
                if not A.is_assigned(to_assign.atom_id):
                    A.add(to_assign, dl, nogood, "propagate")
                    unit_found = True
                    break
        if not unit_found: break
    return "OK", None

def nogood_propagation_wrapper(dl: int, program: Program, delta: Set[Nogood], nabla: Set[Nogood], A: Assignment, checker: UnfoundedSetChecker) -> Tuple[str, Optional[Nogood]]:
    while True:
        status, conflict = unit_propagation(dl, delta | nabla, A)
        if status == "CONFLICT": return "CONFLICT", conflict
        if not program.cyclic_atoms: return "OK", None
        
        U_set = checker.check(A)
        if not U_set: return "OK", None
        
        p = next(iter(U_set))
        loop_nogood_lits = {Literal(p)}
        eb = checker.get_external_bodies(U_set)
        for bid in eb: loop_nogood_lits.add(Literal(-bid))
        loop_nogood = frozenset(loop_nogood_lits)
        
        tracer.log("loop_nogood", {"nogood": [l.id for l in loop_nogood]})
        nabla.add(loop_nogood)

def conflict_analysis(violated: Nogood, A: Assignment, dl: int) -> Tuple[Nogood, int]:
    delta = violated
    tracer.log("analyze_start", {"violated": [l.id for l in delta]})
    while True:
        sigma = A.get_last_assigned_in_nogood(delta, dl)
        # if not sigma:
        #     return delta, 0
        
        # Log resolution step
        tracer.log("analyze_step", {"sigma": sigma.id, "current_delta": [l.id for l in delta]})
        
        sigma_dl = A.get_level(sigma)
        remainder = {A.get_level(r) for r in delta if r != sigma and A.get_level(r) != -1}
        k = max(remainder) if remainder else 0
        
        if k < sigma_dl:
            tracer.log("learned", {"nogood": [l.id for l in delta], "backjump_level": k})
            return delta, k
            
        antecedent = A.get_antecedent(sigma)
        # if not antecedent:
        #     return delta, k
        delta = (delta - {sigma}) | (antecedent - {sigma.complement})

def select_variable(A: Assignment, program: Program) -> Optional[Literal]:
    unassigned = A.get_unassigned_var()
    if unassigned is None: return None
    if unassigned > program.max_atom_id: return Literal(unassigned)
    else: return Literal(-unassigned)

def read_aspif_input() -> str:
    """Read ASPIF input from file (if provided as argument) or stdin."""
    input_text = ""
    # 引数が指定されている場合はファイルから読み込み
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        if not os.path.exists(filepath):
            sys.stderr.write(f"Error: File '{filepath}' not found.\n")
            sys.exit(1)
        with open(filepath, 'r') as f:
            input_text = f.read()
    else:
        # 引数がない場合は標準入力から読み込み
        # 対話実行でない場合のみメッセージを表示しないなどの配慮も可
        if sys.stdin.isatty():
             sys.stderr.write("Reading ASPIF from stdin... (Ctrl+D to finish)\n")
        input_text = sys.stdin.read()

    if not input_text.strip():
        sys.stderr.write("Error: No ASPIF input provided.\n")
        sys.exit(1)
    
    return input_text

def solve(program: Program):
    
    # 1. Completion Nogoods (Delta_Pi)
    delta_pi = program.get_completion_nogoods()
    
    # 2. Integrity Constraint Nogoods
    delta_pi.update(program.constraint_nogoods)
    
    # Statistics: Initial Nogoods
    initial_nogoods_count = len(delta_pi)
            
    all_vars = program.atoms | set(program.id_to_body.keys())
    # Exclude Body IDs that were not registered (i.e., from constraints)
    # all_vars = {v for v in all_vars if v <= program.max_atom_id or v in program.id_to_body}
    
    A = Assignment(all_vars)
    nabla = set()
    dl = 0
    checker = UnfoundedSetChecker(program)
    
    tracer.log("start", {"vars": list(all_vars)})

    # Counters
    steps = 0
    conflicts = 0
    decisions = 0
    sat = False

    while True:
        steps += 1
        status, conflict = nogood_propagation_wrapper(dl, program, delta_pi, nabla, A, checker)
        if status == "CONFLICT":
            conflicts += 1
            if dl == 0:
                tracer.log("result", {"status": "UNSATISFIABLE"})
                sat = False
                break
            learned, k = conflict_analysis(conflict, A, dl)
            nabla.add(learned)
            A.backjump(k)
            dl = k
        else:
            if A.is_total():
                model_atoms = []
                for l in A.literals:
                    if l.is_positive and l.atom_id <= program.max_atom_id:
                        name = program.symbol_table.get(l.atom_id, f"<{l.atom_id}>")
                        model_atoms.append(name)
                tracer.log("result", {"status": "SATISFIABLE", "model": sorted(model_atoms)})
                sat = True
                break
            dec = select_variable(A, program)
            if dec:
                decisions += 1
                dl += 1
                A.add(dec, dl, None, "decide")

    tracer.set_stats({
                "initial_nogoods": initial_nogoods_count,
                "learned_nogoods": len(nabla),
                "total_steps": steps,
                "conflicts": conflicts,
                "decisions": decisions,
                "cyclic_atoms_count": len(program.cyclic_atoms),
                "status": "SATISFIABLE" if sat else "UNSATISFIABLE"
            })
    tracer.dump_json()
    return

if __name__ == "__main__":
    tracer = SolverTracer()

    try:
        input_text = read_aspif_input()
        prog = parse_aspif(input_text)
        solve(prog)
    except Exception as e:
        # JSONが出力されないとフロントエンドが困るのでエラーもJSONにしたい場合があるが、ここでは標準エラーに出す
        sys.stderr.write(f"Solver Error: {str(e)}\n")
        sys.exit(1)