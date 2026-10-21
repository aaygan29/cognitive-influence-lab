/-
  Clique-complex b₁ is NOT monotone under edge addition (unlike graph cyclomatic
  b₁ = E − V + b₀). Witness: the 4-cycle 0–1–2–3–0 has clique b₁ = 1; adding the
  one diagonal {0,2} makes {0,1,2} and {0,2,3} into filled triangles (a disk), so
  clique b₁ drops to 0. Built from first-principles GF(2) homology, no library:
  boundary matrices from the simplex lists, b₁ = nullity ∂₁ − rank ∂₂, the chain
  condition ∂₁∘∂₂ = 0 checked, and the 2-simplices proved to be the 3-cliques.
-/

namespace CliqueBetti

/- ===================== GF(2) linear algebra ===================== -/

/-- XOR of two GF(2) vectors (equal length). -/
def vxor : List Bool → List Bool → List Bool := List.zipWith Bool.xor

/-- Index of the first `true` (the pivot / leading coordinate), if any. -/
def leadIdx : List Bool → Option Nat
  | [] => none
  | true :: _ => some 0
  | false :: t => (leadIdx t).map (· + 1)

/-- Sort key: pivot position, or ∞ for the zero vector. -/
def leadKey (v : List Bool) : Nat := (leadIdx v).getD 1000000

/-- Insert a vector into a basis kept sorted by ascending pivot position. -/
def insertSorted (v : List Bool) : List (List Bool) → List (List Bool)
  | [] => [v]
  | b :: bs => if leadKey v ≤ leadKey b then v :: b :: bs else b :: insertSorted v bs

/-- Reduce `v` against an ascending-pivot basis (one pass suffices: each XOR
    strictly raises `v`'s pivot, and the basis pivots increase too). -/
def reduceVec (basis : List (List Bool)) (v : List Bool) : List Bool :=
  basis.foldl (fun v b =>
    match leadIdx v with
    | none => v
    | some i => if some i == leadIdx b then vxor v b else v) v

def addToBasis (basis : List (List Bool)) (v0 : List Bool) : List (List Bool) :=
  let v := reduceVec basis v0
  match leadIdx v with
  | none => basis
  | some _ => insertSorted v basis

/-- Rank over GF(2) of a list of column vectors, via Gaussian elimination. -/
def rankGF2 (cols : List (List Bool)) : Nat := (cols.foldl addToBasis []).length

/- ===================== The two complexes ===================== -/

def nVerts : Nat := 4

def edgesSmall : List (Nat × Nat) := [(0,1),(1,2),(2,3),(0,3)]
def edgesBig   : List (Nat × Nat) := [(0,1),(1,2),(2,3),(0,3),(0,2)]  -- + diagonal {0,2}

/-- ∂₁ column of an edge {a,b}: indicator over the 4 vertices. -/
def d1col (e : Nat × Nat) : List Bool :=
  (List.range nVerts).map (fun v => v == e.1 || v == e.2)

/-- The three boundary edges of a triangle (vertices ascending → smaller-first). -/
def edgesOf (t : Nat × Nat × Nat) : List (Nat × Nat) :=
  [(t.1, t.2.1), (t.2.1, t.2.2), (t.1, t.2.2)]

/-- ∂₂ column of a triangle: indicator over the ambient edge list `es`. -/
def d2col (es : List (Nat × Nat)) (t : Nat × Nat × Nat) : List Bool :=
  es.map (fun e => (edgesOf t).contains e)

/- --- The 2-simplices are DERIVED as the 3-cliques (flag complex), not chosen. --- -/

def isEdge (es : List (Nat × Nat)) (a b : Nat) : Bool :=
  es.contains (Nat.min a b, Nat.max a b)

/-- All 3-cliques {a<b<c} of the graph given by edge list `es`. -/
def cliques3 (es : List (Nat × Nat)) : List (Nat × Nat × Nat) :=
  (List.range nVerts).flatMap (fun a =>
    (List.range nVerts).flatMap (fun b =>
      (List.range nVerts).filterMap (fun c =>
        if a < b && b < c && isEdge es a b && isEdge es b c && isEdge es a c
        then some (a, b, c) else none)))

/-- Sanity: the flag complex of the 4-cycle has NO triangles; adding {0,2}
    creates exactly {0,1,2} and {0,2,3}. -/
example : cliques3 edgesSmall = [] := by decide
example : cliques3 edgesBig = [(0,1,2),(0,2,3)] := by decide

/- Boundary matrices, built from the (clique-derived) simplex lists. -/
def D1small : List (List Bool) := edgesSmall.map d1col
def D1big   : List (List Bool) := edgesBig.map d1col
def D2small : List (List Bool) := (cliques3 edgesSmall).map (d2col edgesSmall)
def D2big   : List (List Bool) := (cliques3 edgesBig).map (d2col edgesBig)

/- --- Chain-complex sanity: ∂₁ ∘ ∂₂ = 0, so H₁ is well defined. --- -/

def zeroVec (n : Nat) : List Bool := List.replicate n false

/-- Apply ∂₁ to an edge-chain: XOR the vertex-columns of its selected edges. -/
def applyD1 (es : List (Nat × Nat)) (chain : List Bool) : List Bool :=
  (List.zip es chain).foldl
    (fun acc ec => if ec.2 then vxor acc (d1col ec.1) else acc) (zeroVec nVerts)

def chainOK (es : List (Nat × Nat)) (ts : List (Nat × Nat × Nat)) : Bool :=
  (ts.map (d2col es)).all (fun col => applyD1 es col == zeroVec nVerts)

example : chainOK edgesSmall (cliques3 edgesSmall) = true := by decide
example : chainOK edgesBig   (cliques3 edgesBig)   = true := by decide

/- ===================== Betti numbers ===================== -/

/-- b₀ = (#vertices) − rank ∂₁   (number of connected components). -/
def b0 (es : List (Nat × Nat)) : Nat := nVerts - rankGF2 (es.map d1col)

/-- b₁ = nullity ∂₁ − rank ∂₂ = (#edges − rank ∂₁) − rank ∂₂. -/
def b1 (es : List (Nat × Nat)) (ts : List (Nat × Nat × Nat)) : Nat :=
  (es.length - rankGF2 (es.map d1col)) - rankGF2 (ts.map (d2col es))

/-- Both complexes are connected (b₀ = 1), so the effect below is about loops,
    not fragmentation. -/
example : b0 edgesSmall = 1 := by decide
example : b0 edgesBig   = 1 := by decide

/-- The 4-cycle's clique complex has one loop. -/
example : b1 edgesSmall (cliques3 edgesSmall) = 1 := by decide
/-- After adding the single diagonal edge {0,2}, the loop is filled: zero loops. -/
example : b1 edgesBig (cliques3 edgesBig) = 0 := by decide

/- ===================== Main result ===================== -/

/-- Adding the single edge {0,2} to the 4-cycle (a) keeps every existing edge,
    (b) is a genuine new edge, (c) preserves connectivity (b₀ = 1 throughout),
    yet (d) DECREASES the clique-complex loop count b₁ from 1 to 0.
    Hence b₁ of the clique complex is NOT monotone under edge addition. -/
theorem clique_b1_not_monotone :
    edgesSmall.all (fun e => edgesBig.contains e) = true          -- small ⊆ big
    ∧ edgesSmall.contains (0,2) = false ∧ edgesBig.contains (0,2) = true
    ∧ b0 edgesSmall = 1 ∧ b0 edgesBig = 1
    ∧ b1 edgesSmall (cliques3 edgesSmall) = 1
    ∧ b1 edgesBig (cliques3 edgesBig) = 0 := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;> decide

end CliqueBetti

-- Axiom audit: should be `propext` only (no `sorry`, no compiler trust).
#print axioms CliqueBetti.clique_b1_not_monotone
