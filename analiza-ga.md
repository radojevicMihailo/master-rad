# Detaljna analiza `genetic-algorithm.py`

Ovaj dokument analizira svaki segment koda linijski, uz objašnjenje pojmova iz teorije genetskih algoritama (GA) i kombinatorne optimizacije.

---

## 0. Kontekst problema

### Matematička formulacija

Problem je oblik **Generalizovanog dodeljivanja (Generalized Assignment Problem, GAP)** sa proširenjem na količine (integer knapsack po mašini, sa globalnim limitom tražnje):

Neka je:
- $S$ = skup servisa, $|S| = m$
- $C$ = skup računara, $|C| = n$
- $t_{ij}$ = vreme izvršavanja jedinice servisa $i$ na računaru $j$
- $p_{ij}$ = zarada po jedinici
- $d_i$ = tražnja za servisom $i$
- $T$ = maksimalno radno vreme računara (2880 min)
- $x_{ij} \in \mathbb{Z}_{\geq 0}$ = broj jedinica servisa $i$ izvršenih na računaru $j$

**Cilj:**
$$\max \sum_{i=1}^{m} \sum_{j=1}^{n} p_{ij} \cdot x_{ij}$$

**Ograničenja:**
$$\sum_{i=1}^{m} t_{ij} \cdot x_{ij} \leq T \quad \forall j \in C \quad \text{(vreme po računaru)}$$
$$\sum_{j=1}^{n} x_{ij} \leq d_i \quad \forall i \in S \quad \text{(tražnja po servisu)}$$
$$x_{ij} \geq 0, \quad x_{ij} \in \mathbb{Z}$$

Problem je **NP-hard** (redukuje se na multi-dimensional knapsack). Egzaktni solveri (Gurobi) rade za male instance; GA daje aproksimacije za velike.

### Mapiranje na GA

| GA pojam | Konkretno u ovom kodu |
|----------|-----------------------|
| Individua / hromozom | `np.ndarray` oblika `(NUM_SERVICES, NUM_COMPUTERS)` `int32` |
| Gen | Jedna ćelija `chromosome[i, j]` — broj jedinica |
| Alele | Celi brojevi iz $[0, \min(d_i, \lfloor T/t_{ij} \rfloor)]$ |
| Populacija | Lista od `POPULATION_SIZE` matrica |
| Fenotip / genotip | Ovde identični — direktno kodiranje (nema dekoderske funkcije) |
| Fitness | Ukupna zarada |
| Generacija | Jedan prolaz kroz selekciju + reprodukciju |

**Direktno kodiranje** znači da je hromozom već validna alokacija (posle repair-a). Alternativa bi bila permutacijsko kodiranje + dekoder, ali to dodaje kompleksnost.

---

## 1. Uvoz biblioteka (linije 1–8)

```python
import csv, random, os, statistics, time
from datetime import datetime
import numpy as np
```

- `csv` / string split — parsiranje ulaznih fajlova.
- `random` — Python-ov PRNG (Mersenne Twister). Koristi se za turnir (`random.sample`), verovatnoće (`random.random`), odluke (`random.randint`).
- `numpy` — vektorizovane operacije. Ključno za performanse: množenje matrica, suma po osi, maskiranje.
- `statistics` — medijana, prosek, std dev za multi-run analizu.
- `time` / `datetime` — merenje izvršenja i timestampovanje fajlova rezultata.

**Zašto dva PRNG-a (`random` + `np.random`):** Python `random` i NumPy imaju nezavisne generatore. Za reproduktivnost oba se seed-uju (`random.seed` + `np.random.seed`, linije 448–449).

---

## 2. Učitavanje podataka (linije 10–99)

### 2.1 Konfiguracija instance

```python
PROBLEM_SIZE = "10x10"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"podaci-{PROBLEM_SIZE}")
```

Instanca definisana stringom `"10x10"` (10 servisa × 10 računara). `os.path.dirname(os.path.abspath(__file__))` — direktorijum u kom fajl živi, **neosetljivo na `cwd`** (važno kad se pokreće iz drugog direktorijuma).

### 2.2 Dva formata ulaza

Kod podržava:
- **CSV** (`.csv`, sa header-om i prvom kolonom oznakom) — `load_matrix_csv`, `load_demand_csv`
- **TSV** (`.txt`, tab-separated, sa header-om) — `load_matrix_tsv`, `load_demand_tsv`

Fallback logika u `load_data`:
```python
if os.path.exists(csv_time):
    # CSV
else:
    # TSV
```

Prva kolona se preskače (`row[1:]`) jer sadrži oznake tipa "Servis_1", "Racunar_3" — nije numerička.

### 2.3 Konverzija u NumPy

```python
TIME = np.array(_time_list, dtype=np.int32)
PROFIT = np.array(_profit_list, dtype=np.int32)
DEMAND = np.array(_demand_list, dtype=np.int32)
NUM_SERVICES, NUM_COMPUTERS = TIME.shape
MAX_TIME = 2880
```

**Zašto `int32` a ne default `int64`:**
- Pola memorije
- Cache-friendly (više elemenata u L1)
- Vrednosti staju u 32-bit (vreme < 50000, profit < milijardu)

**`NUM_SERVICES, NUM_COMPUTERS = TIME.shape`** — dimenzije izvedene iz podataka, ne hardkodovane. Isti kod radi za 10x10, 100x100, itd.

**`MAX_TIME = 2880`** = 48 sati × 60 minuta. Fiksno za sve instance (pretpostavka problema).

---

## 3. Hiperparametri GA (linije 103–108)

```python
POPULATION_SIZE = 1000
GENERATIONS = 500
TOURNAMENT_SIZE = 5
CROSSOVER_RATE = 0.85
MUTATION_RATE = 0.15
ELITE_COUNT = 10
```

### 3.1 `POPULATION_SIZE = 1000`

**Veličina populacije** — broj jedinki koje paralelno evoluiraju.

**Tradeoff:**
- Velika populacija → visoka raznovrsnost (diversity), bolje istraživanje prostora, sporije generacije.
- Mala populacija → brza konvergencija, rizik preranog zaključavanja u lokalni optimum (**premature convergence**).

**Pravilo palca:** 10× do 100× dimenzija problema. Za 10×10 = 100 dimenzija, populacija 1000 je na gornjem kraju (konzervativno — garantuje raznovrsnost).

### 3.2 `GENERATIONS = 500`

**Broj iteracija** — koliko evolucijskih ciklusa.

**Kriterijum zaustavljanja:** fiksno. Alternative:
- Konvergencija: zaustavi kad fitness ne raste N generacija
- Ciljni fitness: zaustavi kad se postigne
- Vreme: zaustavi posle t sekundi

Ovde je fiksan broj radi **poredivosti** rezultata između runova.

### 3.3 `TOURNAMENT_SIZE = 5`

**Veličina turnira** u turnirskoj selekciji.

**Teorija:** verovatnoća da najbolja jedinka iz populacije veličine $N$ bude izabrana u turniru veličine $k$:
$$P(\text{best selected}) = 1 - \left(1 - \frac{1}{N}\right)^k \approx \frac{k}{N}$$

**Pritisak selekcije** raste sa $k$:
- $k = 1$ → čisti random (nema pritiska)
- $k = 2$ → blag pritisak, velika raznovrsnost
- $k = 5$ → umeren pritisak (ovde)
- $k = N$ → samo najbolji (deterministic, truncation selection)

Izbor 5 iz populacije 1000 = $5/1000 = 0.5\%$ šansa da najbolji bude biran — blag pritisak, čuva raznovrsnost.

### 3.4 `CROSSOVER_RATE = 0.85`

**Verovatnoća ukrštanja.** Sa $p_c = 0.85$, 85% parova roditelja proizvode rekombinante; 15% se kopiraju nepromenjeni.

**Tipičan opseg:** 0.6–0.95. Niže → spora konvergencija; više → rizik preranog gubitka raznovrsnosti.

### 3.5 `MUTATION_RATE = 0.15`

**Verovatnoća mutacije.**

**Napomena:** u kodu se ovaj `MUTATION_RATE` koristi kao **očekivani broj mutiranih kolona po redu**, ne po celoj jedinki:
```python
per_cell_probability = MUTATION_RATE / NUM_COMPUTERS
```

Dakle efektivna verovatnoća po ćeliji = $0.15 / 10 = 0.015$ za 10x10. Očekivani broj mutiranih ćelija po matrici = $0.015 \times 100 = 1.5$. Razumno — De Jongov savet je $1/L$ po bitu (~$1/100 = 0.01$).

**Svrha mutacije:**
- Održavanje raznovrsnosti
- Beg iz lokalnih optimuma
- Uvođenje alela koji su se izgubili iz populacije

### 3.6 `ELITE_COUNT = 10`

**Elitizam** — top 10 jedinki kopira se direktno u narednu generaciju, netaknuto.

**Zašto:**
- Garantuje **monotonost**: best-fitness(gen $n+1$) ≥ best-fitness(gen $n$)
- Sprečava destrukciju dobrih rešenja mutacijom/crossover-om

**Rizik:** preveliki `ELITE_COUNT` → prerana konvergencija (elita guši raznovrsnost). Pravilo: 1–5% populacije. Ovde 10/1000 = 1% — konzervativno.

---

## 4. Pretkomputirani helperi (linije 120–138)

### 4.1 `PROFIT_PER_MINUTE`

```python
PROFIT_PER_MINUTE = PROFIT.astype(np.float64) / TIME.astype(np.float64)
```

**Heuristička metrika:** dinar po minutu za svaki par $(i, j)$. Ovo je osnovna **density** metrika knapsack problema:
$$\rho_{ij} = \frac{p_{ij}}{t_{ij}}$$

Pohlepni algoritmi za knapsack biraju po opadajućoj $\rho$ — daje aproksimaciju sa faktorom 2 u klasičnom knapsack-u.

**Konverzija u float64:** deljenje celih može da skrati (integer division). Eksplicitan cast = korektna realna podela.

### 4.2 `MAX_UNITS`

```python
MAX_UNITS = (MAX_TIME // TIME).astype(np.int32)
```

Gornja granica po ćeliji **čisto vremenski**: koliko jedinica servisa $i$ može da stane na računar $j$ ako ne bude ničeg drugog.

Koristi se u:
- `init_random` — gornja granica random vrednosti
- `mutate` — clipping posle point mutacije

**Zašto ne `DEMAND`:** tražnja je po servisu (red), ali ćelija takođe ograničena vremenom — pravi limit je $\min(d_i, \lfloor T/t_{ij} \rfloor)$.

### 4.3 `SORTED_PAIRS`

```python
_flat_indices = np.argsort(PROFIT_PER_MINUTE.ravel())[::-1]
SORTED_PAIRS = list(zip(*np.unravel_index(_flat_indices, PROFIT_PER_MINUTE.shape)))
```

Dekonstrukcija:
1. `.ravel()` — spljošti 2D matricu u 1D vektor (row-major).
2. `argsort(...)` — indeksi koji bi sortirali rastuće.
3. `[::-1]` — obrni → opadajuće.
4. `unravel_index(..., shape)` — konvertuj 1D indekse nazad u par (row, col).
5. `zip(*...)` — spoji u listu tuples `(servis, računar)`.

**Rezultat:** svih $m \times n$ parova sortirano od najefikasnijeg ka najneefikasnijem. Koristi se u `init_greedy` za pohlepno dodeljivanje.

**Kompleksnost:** $O(mn \log(mn))$ — radi se **jednom**, globalno.

### 4.4 `SERVICES_BY_RATIO_PER_COMPUTER` i `SERVICES_BEST_FIRST_PER_COMPUTER`

```python
SERVICES_BY_RATIO_PER_COMPUTER = [
    np.argsort(PROFIT_PER_MINUTE[:, computer]).tolist()
    for computer in range(NUM_COMPUTERS)
]
SERVICES_BEST_FIRST_PER_COMPUTER = [
    np.argsort(PROFIT_PER_MINUTE[:, computer])[::-1].tolist()
    for computer in range(NUM_COMPUTERS)
]
```

Po-kolona (po računaru) sortiranje servisa:
- **worst-first** — za repair pri uklanjanju viška (izbacuj najmanje profitabilne prvo)
- **best-first** — za repair pri popunjavanju slobodnog vremena (dodaj najprofitabilnije prvo)

Konverzija `.tolist()` — Python liste su brže za iteraciju u Python petlji nego NumPy nizovi (nema overhead-a za indeksiranje svaki put).

**Memorija:** $O(n \cdot m)$ ukupno — trivijalno.

---

## 5. Repair operator (linije 143–185)

**Ideja:** posle crossover/mutacije, hromozom može prekršiti ograničenja (prevelika alokacija, negativne vrednosti). Repair funkcija deterministički popravlja jedinku u validnu.

### 5.1 Filozofski izbor: Lamarkijanski vs Baldwinov GA

- **Lamarkijanski:** repair menja genom trajno (popravljena verzija se vraća u populaciju). **Ovde korišćeno.**
- **Baldwinov:** fitness se računa na popravljenoj verziji, ali original ostaje u populaciji.

Lamarkijanski brže konvergira, ali može smanjiti raznovrsnost.

### 5.2 Alternativni pristupi ograničenjima

1. **Penalizacija u fitness:** $f'(x) = f(x) - \lambda \cdot \text{violation}(x)$. Problem: biranje $\lambda$, može dozvoliti infeasible rešenja da prežive.
2. **Death penalty:** infeasible = 0 fitness. Problem: gubitak korisnih genotipova.
3. **Repair (ovde):** skupo po iteraciji, ali garantuje validnost.

### 5.3 Korak 1 — Clip negativa

```python
np.clip(chromosome, 0, None, out=chromosome)
```

Mutacija može dati negativne delte (`-delta_max`). Clip na 0. `out=chromosome` = in-place, bez alokacije.

### 5.4 Korak 2 — Tražnja po vrstama

```python
row_totals = chromosome.sum(axis=1)
over_demand = row_totals > DEMAND
if over_demand.any():
    ratios = DEMAND[over_demand].astype(np.float64) / row_totals[over_demand]
    chromosome[over_demand] = (chromosome[over_demand] * ratios[:, np.newaxis]).astype(np.int32)
```

**Logika:** ako red $i$ prekoračuje $d_i$, **proporcionalno skaliraj sve ćelije u tom redu nadole** da suma bude $d_i$.

**Zašto proporcionalno (a ne npr. oduzimanje od najgoreg):**
- Čuva relativnu distribuciju (strukturu rešenja koja je nastala crossover-om/mutacijom)
- Vektorizovano — jednim operacijom popravi sve preopterećene redove

**Numerička sitnica:** `astype(np.int32)` radi floor. Suma posle može biti **manja** od $d_i$ (gubimo malo kapaciteta). To se kompenzuje u koraku 4 (fill).

**`ratios[:, np.newaxis]`** — broadcast: pretvara 1D vektor (jedan ratio po redu) u kolonu, tako da množenje radi po elementima preko kolona.

### 5.5 Korak 3 — Vreme po kolonama

```python
for computer in range(NUM_COMPUTERS):
    used = int(np.sum(TIME[:, computer] * chromosome[:, computer]))
    if used > MAX_TIME:
        for service in SERVICES_BY_RATIO_PER_COMPUTER[computer]:
            if used <= MAX_TIME:
                break
            if chromosome[service, computer] > 0:
                excess = used - MAX_TIME
                units_to_remove = min(
                    int(chromosome[service, computer]),
                    (excess + int(TIME[service, computer]) - 1) // int(TIME[service, computer])
                )
                chromosome[service, computer] -= units_to_remove
                used -= units_to_remove * int(TIME[service, computer])
```

**Logika:** za svaki računar, ako ukupno vreme prekoračuje $T$:
1. Iteriraj servise sortirane od najmanje efikasnih (`SERVICES_BY_RATIO_PER_COMPUTER` = worst-first)
2. Uklanjaj jedinice jedne po jedne dok ne uđeš u limit

**Ključna formula:**
```python
units_to_remove = min(
    ch[s, c],
    ceil(excess / time)
)
```

Gde `ceil(excess / time)` = minimalan broj jedinica koji treba ukloniti da pokrije višak. `(x + y - 1) // y` je integer ceil (trik bez floats).

**Zašto worst-first:** pohlepno čuva najisplativije parove. Ako moraš da žrtvuješ kapacitet, žrtvuj najjeftiniji.

**Kompleksnost:** $O(n \cdot m)$ u najgorem slučaju. Za 10×10 trivijalno, za 100×100 zamere.

### 5.6 Korak 4 — Fill leftover

```python
for computer in range(NUM_COMPUTERS):
    used = int(np.sum(TIME[:, computer] * chromosome[:, computer]))
    remaining = MAX_TIME - used
    for service in SERVICES_BEST_FIRST_PER_COMPUTER[computer]:
        if remaining < TIME[service, computer]:
            continue
        demand_left = int(DEMAND[service]) - int(chromosome[service].sum())
        if demand_left <= 0:
            continue
        add = min(demand_left, remaining // int(TIME[service, computer]))
        if add > 0:
            chromosome[service, computer] += add
            remaining -= add * int(TIME[service, computer])
        if remaining < 5:
            break
```

**Logika:** preostalo vreme = $T - \text{used}$. Pohlepno dodaj jedinice najprofitabilnijih servisa koji:
- Staju u preostalo vreme (`remaining >= TIME[s,c]`)
- Imaju još neispunjenu tražnju (`demand_left > 0`)

**`remaining < 5` break:** mikro-optimizacija. Ako je preostalo <5 min, teško da bilo šta staje (minimalno vreme servisa tipično ≥ 5 min). Izbegava suvišne iteracije.

**Zašto je ovaj korak kritičan:**
- Korak 2 (proporcionalno skaliranje) truncate-uje, ostavlja "rupe"
- Korak 3 uklanja — ostavlja slobodno vreme
- Bez koraka 4, repair bi davao lošija rešenja nego nužno

Ovo je zapravo **lokalna pohlepna poboljšavanje** — čini repair ne samo popravkom, već i poboljšivačem.

---

## 6. Fitness funkcija (linije 190–191)

```python
def fitness(chromosome):
    return int(np.sum(PROFIT * chromosome))
```

**Fitness = ukupna zarada** = $\sum_{ij} p_{ij} x_{ij}$.

**Karakteristike:**
- Nema penalizacije (repair garantuje validnost)
- Vektorizovana — $O(mn)$ ali jedan NumPy poziv
- Vraća `int` — stabilno za sortiranje, hashing, printovanje sa zarezima

**Zašto NE penalizacija:** repair model je "čišći" — fitness je interpretabilno direktno kao ciljna funkcija problema.

---

## 7. Inicijalizacija populacije (linije 196–226)

### 7.1 `init_greedy` (definisan, ali ne korišćen direktno)

```python
def init_greedy():
    chromosome = np.zeros((NUM_SERVICES, NUM_COMPUTERS), dtype=np.int32)
    remaining_demand = DEMAND.copy()
    remaining_time = np.full(NUM_COMPUTERS, MAX_TIME, dtype=np.int32)

    for service, computer in SORTED_PAIRS:
        if remaining_demand[service] <= 0 or remaining_time[computer] < TIME[service, computer]:
            continue
        max_by_time = remaining_time[computer] // TIME[service, computer]
        max_by_demand = remaining_demand[service]
        value = min(int(max_by_time), int(max_by_demand))
        if value > 0:
            value = max(0, value - random.randint(0, value // 3 + 1))
            chromosome[service, computer] = value
            remaining_demand[service] -= value
            remaining_time[computer] -= value * TIME[service, computer]

    return repair(chromosome)
```

**Pohlepna konstruktivna heuristika.** Ide kroz sve parove po opadajućem $\rho_{ij}$ i dodaje maksimum dozvoljen.

**Ključni detalj — stohastičnost:**
```python
value = max(0, value - random.randint(0, value // 3 + 1))
```
Umesto da uzme pun max, uzima `value - random(0, value/3)` → 66–100% max. **Svrha:** ako bi bio deterministic, svaka jedinka bi bila identična. Ovo ubacuje raznovrsnost.

**Zašto nije korišćen u populaciji:** `init_population` koristi `init_random`. Greedy bi dao jako biased start (sve blizu istog lokalnog optimuma). Može se mešati: npr. 10% greedy + 90% random za "seeded" populaciju.

### 7.2 `init_random`

```python
def init_random():
    upper_bounds = np.minimum(DEMAND[:, np.newaxis], MAX_UNITS)
    chromosome = np.zeros((NUM_SERVICES, NUM_COMPUTERS), dtype=np.int32)
    for service in range(NUM_SERVICES):
        for computer in range(NUM_COMPUTERS):
            chromosome[service, computer] = random.randint(0, int(upper_bounds[service, computer]))
    return repair(chromosome)
```

**Uniformni random u pametnim granicama.**

`upper_bounds[i,j] = min(d_i, floor(T/t_{ij}))` — realna maksimalna vrednost ćelije. `DEMAND[:, np.newaxis]` širi 1D vektor (dužine $m$) u kolonu matrice $m \times 1$, pa broadcast sa `MAX_UNITS` (matrica $m \times n$).

**Zašto ne direktno `np.random.randint(0, upper_bounds)`:** NumPy prihvata samo skalar high u `randint`, ne matricu. Mora element-wise. (Mogla bi se koristiti `np.random.randint(upper_bounds+1)` sa matricom — TODO optimizacija.)

**Posle random inicijalizacije velika verovatnoća narušenih ograničenja → repair** je nužan.

### 7.3 `init_population`

```python
def init_population(size):
    return [init_random() for _ in range(size)]
```

Samo random. Za 1000 jedinki × 100 ćelija × repair = najskuplji korak pored glavne petlje, ali se radi jednom.

---

## 8. Selekcija — Turnir (linije 231–234)

```python
def tournament_select(population, fitnesses, tournament_size=TOURNAMENT_SIZE):
    indices = random.sample(range(len(population)), tournament_size)
    best_index = max(indices, key=lambda idx: fitnesses[idx])
    return population[best_index].copy()
```

### 8.1 Algoritam

1. Izvuci $k$ različitih indeksa (bez ponavljanja — `random.sample`).
2. Među njima pronađi onog sa maksimalnim fitness-om.
3. Vrati kopiju te jedinke.

### 8.2 Alternative i zašto turnir

| Metoda | Kako | Mana |
|--------|------|------|
| **Roulette wheel** | $P(i) = f_i / \sum f$ | Loše skaliranje; ako jedan $f$ dominira, monopol |
| **Rank** | $P(i)$ zavisi od ranga, ne vrednosti | Treba sortiranje svake generacije, $O(N \log N)$ |
| **Truncation** | Top 50% → roditelji | Ekstremna erozija raznovrsnosti |
| **Turnir** | $k$ random, pobednik | Nezavisan od skale fitnessa, konstantan pritisak, $O(k)$ po selekciji |

### 8.3 Zašto `.copy()`

Ako se vrati referenca, mutacija deteta menja roditelja u populaciji. Kopija je nužna pre modifikacije. NumPy `.copy()` je plitak ali matrice su "plitke" po prirodi (samo int32).

### 8.4 Zašto `random.sample` a ne `random.choices`

- `sample` — bez ponavljanja. Turnir sa ponavljanjima bio bi iskrivljen (ista jedinka više puta).
- `choices` — sa ponavljanjem.

---

## 9. Crossover — Uniformni (linije 239–247)

```python
def uniform_crossover(parent1, parent2):
    if random.random() > CROSSOVER_RATE:
        return parent1.copy(), parent2.copy()

    mask = np.random.randint(0, 2, size=(NUM_SERVICES, NUM_COMPUTERS), dtype=np.bool_)
    child1 = np.where(mask, parent1, parent2)
    child2 = np.where(mask, parent2, parent1)

    return repair(child1), repair(child2)
```

### 9.1 Tipovi crossover-a

| Tip | Opis | Prikladno za |
|-----|------|--------------|
| **One-point** | Seci u jednoj tački, zameni repove | Sekvencijalni geni sa pozicijskom zavisnošću |
| **Two-point** | Dve tačke reza | Slično, manje disruption |
| **Uniform (ovde)** | Svaki gen nezavisno baca novčić | Nezavisni geni (naš slučaj) |
| **Arithmetic** | $c = \alpha p_1 + (1-\alpha) p_2$ | Realne vrednosti |
| **Block / row / column** | Razmenjuju se celi blokovi | Strukturni problemi |

**Zašto uniformni ovde:**
- Geni (ćelije) nisu pozicijski povezani — ćelija (3,5) nema veze sa (3,6) osim preko row/column constraints
- Maksimalna rekombinacija

**Potencijalna alternativa — row crossover:** razmeni cele redove (servise). Može biti bolji jer red odgovara jednoj logičkoj jedinici (tražnja po servisu). Vredi eksperimentisati.

### 9.2 Implementacija

```python
mask = np.random.randint(0, 2, size=..., dtype=np.bool_)
child1 = np.where(mask, parent1, parent2)
child2 = np.where(mask, parent2, parent1)
```

`np.where(mask, a, b)` — ternarna select: `mask=True → a, False → b`. Child2 je **komplementaran** childu1 (svaki gen od suprotnog roditelja). Maksimalno očuvanje materijala — dva deteta sadrže svih $2mn$ gena iz oba roditelja.

### 9.3 Probabilistic skip

```python
if random.random() > CROSSOVER_RATE:
    return parent1.copy(), parent2.copy()
```

Sa verovatnoćom $1-p_c$ = 0.15, ne ukršta — samo kopira roditelje. To propušta "stabilne" jedinke kroz generaciju bez disrupcije (sličan efekat elitizmu na nivou para).

### 9.4 Repair posle

Kombinacija dve validne jedinke može biti nevalidna (npr. parent1 koristi računar 3 intenzivno, parent2 takođe — maska može uzeti obe strane za gomilu servisa, prekoračenje vremena). Repair obavezan.

---

## 10. Mutacija (linije 252–277)

**Tri operatora u jednoj funkciji** — hibridna mutacija.

### 10.1 Operator 1 — Point mutation (tačkasta)

```python
per_cell_probability = MUTATION_RATE / NUM_COMPUTERS
mutation_mask = np.random.random((NUM_SERVICES, NUM_COMPUTERS)) < per_cell_probability
if mutation_mask.any():
    deltas_max = np.maximum(1, chromosome // 3)
    random_deltas = (np.random.random(chromosome.shape) * (2 * deltas_max + 1) - deltas_max).astype(np.int32)
    chromosome[mutation_mask] += random_deltas[mutation_mask]
    np.clip(chromosome, 0, MAX_UNITS, out=chromosome)
```

**Korak po korak:**

1. **Verovatnoća po ćeliji:**
   ```python
   per_cell_probability = MUTATION_RATE / NUM_COMPUTERS = 0.15 / 10 = 0.015
   ```
   Očekivani broj mutacija po redu ≈ 0.15 (otud ime `MUTATION_RATE`). Prosta statistika: $E[\text{mutated cells}] = mn \cdot p = 100 \cdot 0.015 = 1.5$ za 10×10.

2. **Generisanje maske:**
   ```python
   mutation_mask = np.random.random(...) < per_cell_probability
   ```
   Uniform [0,1) po ćeliji, maska True gde je vrednost manja.

3. **Adaptivna delta:**
   ```python
   deltas_max = np.maximum(1, chromosome // 3)
   ```
   Veća ćelija → veća mogućnost promene. Ćelija 30 → $\Delta \in [-10, 10]$; ćelija 3 → $\Delta \in [-1, 1]$ (jer max(1, 1)=1). Ćelija 0 → $\Delta \in [-1, 1]$.

   **Zašto adaptivno:** fiksna delta (npr. $\pm 5$) bi na malim ćelijama uništila (npr. ćelija 2 → -3 ≡ 0), a na velikim jedva promenila. Adaptivnost skalira mutaciju srazmerno.

4. **Random delta u $[-\Delta_{max}, +\Delta_{max}]$:**
   ```python
   random_deltas = random[0,1) * (2*deltas_max + 1) - deltas_max
   ```
   Matematika: $[0, 1) \to [0, 2\Delta+1) \to [-\Delta, \Delta+1) \to$ integer cast $\to [-\Delta, \Delta]$.

   Uniform raspodela delte — neutralna (ne biasira ni na povećanje ni na smanjenje).

5. **Primena samo gde je maska True:**
   ```python
   chromosome[mutation_mask] += random_deltas[mutation_mask]
   ```

6. **Clip:**
   ```python
   np.clip(chromosome, 0, MAX_UNITS, out=chromosome)
   ```
   Vrati u validni opseg po ćeliji.

### 10.2 Operator 2 — Column swap (5%)

```python
if random.random() < 0.05:
    computer1, computer2 = random.sample(range(NUM_COMPUTERS), 2)
    chromosome[:, [computer1, computer2]] = chromosome[:, [computer2, computer1]]
```

**Makro-mutacija:** zameni cele alokacije dva računara.

**Zašto korisno:** ako su dva računara slično efikasna ali ne identična, swap može otkriti bolju konfiguraciju bez postepenog otkrivanja tačkastom mutacijom (bilo bi potrebno $O(m)$ mutacija da se postigne isti efekat).

**Sprečava plateau:** kad tačkasta mutacija ne pomera jer je raspored lokalno optimalan, swap drastično menja topologiju.

**5% = retko:** disruptivna je, ne sme biti česta.

### 10.3 Operator 3 — Row redistribution (5%)

```python
if random.random() < 0.05:
    service = random.randint(0, NUM_SERVICES - 1)
    total = int(chromosome[service].sum())
    if total > 0:
        weights = PROFIT_PER_MINUTE[service]
        weights_sum = weights.sum()
        chromosome[service] = (total * weights / weights_sum).astype(np.int32)
```

**Pametna mutacija vođena heuristikom:** uzme ukupnu alokaciju servisa $i$ i preraspodeli po računarima proporcionalno efikasnosti.

Matematika:
$$x'_{ij} = \lfloor \text{total}_i \cdot \frac{\rho_{ij}}{\sum_k \rho_{ik}} \rfloor$$

**Zašto:** ako je servis "slučajno" dodeljen neefikasnim računarima, ovaj operator ga pomera ka efikasnijim bez da menja ukupnu količinu.

**To je hibrid GA + pohlepna heuristika.** U čistom GA mutacija je random; ovde usmerena ka dobrom rešenju. Tip **memetičkog algoritma** (GA + lokalno poboljšanje).

**Rizik:** smanjuje raznovrsnost ka pohlepnom rešenju. 5% je kompromis.

### 10.4 Finalni repair

```python
return repair(chromosome)
```

Nužno — mutacija može prekršiti row sum (posle point mut.), column time (posle column swap ako računari imaju različite kapacitete — ovde ne, ali general pattern).

---

## 11. Glavna GA petlja (linije 282–332)

```python
def run_ga():
    print("\nInicijalizacija populacije...")
    population = init_population(POPULATION_SIZE)
    best_ever = None
    best_fitness_ever = 0
    stagnation_counter = 0

    for generation in range(GENERATIONS):
        fitnesses = [fitness(chromosome) for chromosome in population]

        generation_best_index = max(range(len(fitnesses)), key=lambda idx: fitnesses[idx])
        generation_best_fitness = fitnesses[generation_best_index]

        if generation_best_fitness > best_fitness_ever:
            best_fitness_ever = generation_best_fitness
            best_ever = population[generation_best_index].copy()
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        if generation % 25 == 0:
            average_fitness = sum(fitnesses) / len(fitnesses)
            print(f"Gen {generation:4d} | Best: {generation_best_fitness:,} | ...")

        if stagnation_counter > 0 and stagnation_counter % 100 == 0:
            sorted_indices = sorted(range(len(fitnesses)), key=lambda idx: fitnesses[idx])
            num_replace = POPULATION_SIZE // 5
            for idx in sorted_indices[:num_replace]:
                population[idx] = init_random()

        elite_indices = sorted(range(len(fitnesses)), key=lambda idx: fitnesses[idx], reverse=True)[:ELITE_COUNT]
        new_population = [population[idx].copy() for idx in elite_indices]

        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_select(population, fitnesses)
            parent2 = tournament_select(population, fitnesses)
            child1, child2 = uniform_crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)
            new_population.append(child1)
            if len(new_population) < POPULATION_SIZE:
                new_population.append(child2)

        population = new_population

    return best_ever, best_fitness_ever
```

### 11.1 Struktura po generaciji

1. **Evaluacija:** fitness za celu populaciju. $O(N \cdot mn)$.
2. **Praćenje best-ever:** globalno maksimum. Elitizam većinski pokriva, ali sigurnosna mreža (u slučaju baga u elitizmu).
3. **Logging svakih 25 generacija:** best, avg, all-time. Ne na svakoj — buka + I/O sporije.
4. **Diversity injection na stagnaciji.**
5. **Elitizam.**
6. **Reprodukcija.**

### 11.2 Stagnation & Diversity injection

```python
if stagnation_counter > 0 and stagnation_counter % 100 == 0:
    sorted_indices = sorted(range(len(fitnesses)), key=lambda idx: fitnesses[idx])
    num_replace = POPULATION_SIZE // 5
    for idx in sorted_indices[:num_replace]:
        population[idx] = init_random()
```

**Logika:** ako 100 generacija nije bilo napretka, zameni najgorih 20% populacije novim random jedinkama.

**Šta je stagnacija:** `best_fitness_ever` se ne povećava. Može značiti:
- Stigao globalni optimum (retko prepoznatljivo)
- Zaglavljen u lokalnom optimumu (najčešće)
- Populacija konvergirala na istu jedinku (genetska monotonija)

**Random restart deo populacije** = **niche preservation / crowding** alternativa. Injektuje svežu genetiku bez rušenja elite.

**Zašto svakih 100:** dovoljno da se GA oporavi pre nove injekcije. Sa 500 generacija, do 4 injekcije max.

**Zašto baš 20%:** empirijski kompromis. Manje = nedovoljno impact. Više = gubitak progresa.

### 11.3 Elitizam

```python
elite_indices = sorted(range(len(fitnesses)), key=lambda idx: fitnesses[idx], reverse=True)[:ELITE_COUNT]
new_population = [population[idx].copy() for idx in elite_indices]
```

Top 10 kopiraju se u novu generaciju netaknuti. **Monotonost garancija.**

**`.copy()` obavezno** — iako ovi nisu prolaze kroz mutate, sledećih generacija će biti roditelji, a njihova kopija bi bila menjana ako ovde ne bude deep copy.

### 11.4 Reprodukcija

```python
while len(new_population) < POPULATION_SIZE:
    parent1 = tournament_select(...)
    parent2 = tournament_select(...)
    child1, child2 = uniform_crossover(parent1, parent2)
    child1 = mutate(child1)
    child2 = mutate(child2)
    new_population.append(child1)
    if len(new_population) < POPULATION_SIZE:
        new_population.append(child2)
```

**Pipeline:** selekcija → crossover → mutacija. Standardna GA šema.

**Dva deteta po paru** — efikasno (jedan crossover daje dva rezultata umesto jednog). Inner `if` štiti od overflow-a ako je `POPULATION_SIZE` neparno posle elitizma.

**Selekcija iz stare populacije `population`** (ne iz `new_population`) — **generacijski GA**. Alternativa: **steady-state** (pojedinačne zamene u istoj populaciji).

### 11.5 Kompleksnost po generaciji

- Eval: $O(N \cdot mn)$ = $1000 \cdot 100 = 10^5$ op
- Sort za elitizam: $O(N \log N)$
- Reprodukcija: $N/2$ parova, svaki:
  - 2 turnira = $O(k)$
  - Crossover = $O(mn)$
  - Mutacija = $O(mn)$
  - Repair = $O(nm)$ najgore
- Total po generaciji ≈ $O(N \cdot mn)$

Za 500 generacija, 1000 populacije, 100 gena: $5 \times 10^7$ osnovnih operacija. Sa NumPy vektorizacijom — sekundi.

---

## 12. Output funkcije (linije 337–385)

`print_solution` formatira konačno rešenje:

### 12.1 Iskorišćenje računara

```python
time_per_computer = np.sum(TIME * chromosome, axis=0)
```

`axis=0` — suma po redovima (rezultat po kolonama = po računaru).

Prikazuje samo računare sa < 99% iskorišćenja (one "nedostignute"). Cilj analize: **gde je kapacitet ostao neiskorišćen?** Ako je 100%, rešenje maksimalno stiska resurs.

### 12.2 Iskorišćenje tražnje

```python
allocated_per_service = chromosome.sum(axis=1)
services_at_max = int(np.sum(allocated_per_service == DEMAND))
services_unused = int(np.sum(allocated_per_service == 0))
```

- Servisi 100% iskorišćeni (tražnja zadovoljena)
- Servisi 0 (uopšte ne korišćeni — verovatno neprofitabilni)
- Delimično korišćeni

**Interpretacija:** ako je mnogo servisa na 0, problem je vremenski zagušen. Ako je mnogo na max, problem je demand-zagušen.

### 12.3 Top 15 dodela

```python
contributions = PROFIT * chromosome
nonzero = np.argwhere(chromosome > 0)
assignments = [(contribution, s, c, qty, price) for ...]
assignments.sort(reverse=True)
```

Prikazuje pojedinačne dodele (par servis-računar) sortirane po ukupnom doprinosu. **Analitička vrednost:** vidiš koji parovi čine srž profita.

---

## 13. Multi-run orkestracija (linije 390–432, 437–481)

### 13.1 Zašto 10 runova

GA je **stohastičan**. Jedan run nije reprezentativan:
- Random seed pravi različite trajektorije evolucije
- Može zaglaviti u različitim lokalnim optimumima

10 runova sa seedovima 0–9 daje:
- Medijanu — robusan centar raspodele
- Min/max — opseg kvaliteta
- Std dev — stabilnost algoritma

**Medijana, ne prosek:** robusnija na outlier-e. Ako jedan run zaglavi loše, prosek ga vuče; medijana ignoriše.

### 13.2 Reproduktivnost

```python
random.seed(run_index)
np.random.seed(run_index)
```

Seed po run-u = deterministički rezultati. Važno za:
- Debug
- Ponavljanje eksperimenata
- Poređenje hiperparametara (isti seedovi, različit `POPULATION_SIZE`)

**OBA PRNG-a moraju biti seeded** jer su nezavisni.

### 13.3 Čuvanje rezultata

```python
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"ga-{POPULATION_SIZE}-{GENERATIONS}-{ELITE_COUNT}-{NUM_RUNS}runs-{timestamp}.txt"
```

Ime fajla kodira hiperparametre + timestamp → lako pretraživo, nema preklapanja.

Sadržaj: svi hiperparametri, rezultati svih runova, agregirana statistika, vremena.

---

## 14. Performansne karakteristike

### 14.1 Vektorizacija

**Ključni faktor brzine.** Svaka NumPy operacija ide kroz C loop umesto Python interpretera. Grubo poređenje:
- Python petlja po 10×10 matrici: ~10μs
- NumPy ekvivalent: ~1μs

Multiplikatori se akumuliraju: 500 generacija × 1000 pop × 100 gena = $5 \times 10^7$ touch operacija.

### 14.2 Bottlenecks

1. **Repair funkcija** (linije 156–183) — unutrašnje Python petlje, ne vektorizovane. Za 100×100 postaje značajno.
2. **Mutacija** — većinski vektorizovana, ali hybrid ops (swap, redistribute) su Python.
3. **`chromosome.copy()`** pozivi — alokacija nove memorije po pozivu. Sa 1000 populacijom × 500 gen = 500K kopija.

### 14.3 Moguće optimizacije

- `@numba.njit` na repair-u
- Pool pre-alociranih matrica (reuse memory)
- Repair samo ako je ograničenje stvarno narušeno (provera pre skupe logike)
- Fitness cache (ako se jedinka ne menja)

---

## 15. Celokupni tok algoritma (dijagram)

```
┌─────────────────────────────────────────────┐
│ Load TIME, PROFIT, DEMAND                   │
│ Precompute PROFIT_PER_MINUTE, SORTED_PAIRS  │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│ For run in 0..9:                            │
│   seed(run)                                 │
│   best, profit = run_ga()                   │
└─────────────────────────────────────────────┘
                    ↓ (run_ga)
┌─────────────────────────────────────────────┐
│ init_population(1000) via init_random       │
│ For each init: random fill → repair         │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│ For gen in 0..499:                          │
│   fitnesses = [fitness(c) for c in pop]     │
│   update best_ever                          │
│   if stagnation % 100: inject diversity     │
│   elite (top 10) → new_pop                  │
│   While new_pop < 1000:                     │
│     p1 = tournament(5)                      │
│     p2 = tournament(5)                      │
│     c1, c2 = uniform_crossover(p1, p2)      │
│     c1 = mutate(c1)                         │
│     c2 = mutate(c2)                         │
│     Each mutate ends with repair            │
│     new_pop += [c1, c2]                     │
│   pop = new_pop                             │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│ Return best_ever                            │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│ Aggregate 10 runs: median, min, max, mean   │
│ print_solution(best_overall)                │
│ save_results() → timestamped .txt           │
└─────────────────────────────────────────────┘
```

---

## 16. Teorijski aspekti — zašto GA uopšte radi

### 16.1 Schema teorema (Holland)

Schema = šablon nad alelima (npr. `*1*0*`). GA "propagira" šeme iznadprosečnog fitnessa. Za uniformni crossover:
$$m(H, t+1) \geq m(H, t) \cdot \frac{f(H)}{\bar{f}} \cdot (1 - p_m)^{o(H)}$$

Gde $o(H)$ = red šeme, $p_m$ = mut. rate. Iznadprosečne šeme se eksponencijalno množe.

### 16.2 Building blocks hipoteza

GA kombinuje **mala** dobra pod-rešenja (building blocks) u veća. Uniformni crossover je agnostičan na pozicije — dobar kad su building blocks rasuti.

### 16.3 Exploration vs Exploitation

| Mehanizam | Exploration | Exploitation |
|-----------|-------------|--------------|
| Mutacija | ✓ | |
| Crossover | ✓ (nove kombinacije) | ✓ (koristi postojeći materijal) |
| Selekcija (turnir) | | ✓ |
| Elitizam | | ✓ (ekstremno) |
| Diversity injection | ✓ (eksplozivno) | |

Dobar GA balansira oba. Ovaj kod: visok CR (0.85) + moderan MR (0.15) + elitizam + stagnation injection = naglasak na exploitation sa safety net za exploration.

---

## 17. Moguća poboljšanja i eksperimenti

1. **Adaptivni parametri:** smanjivati `MUTATION_RATE` kako GA konvergira.
2. **Lokalna pretraga (memetic):** posle repair-a, 2-opt swap.
3. **Hibridna inicijalizacija:** 10% `init_greedy`, 90% `init_random`.
4. **Row / column crossover** umesto uniform — eksperiment.
5. **Parallel evaluation:** `multiprocessing.Pool` na fitness.
6. **Viši `TOURNAMENT_SIZE`** ako prerano stagnira; niži ako se zaglavi.
7. **Replace policy:** elitist + truncation hybrid (top 50% plus elita).
8. **LP-relaksacija kao upper bound:** utvrdi koliko je GA daleko od optimuma.

---

## 18. Sažetak parametara vs njihov efekat

| Parametar | Trenutno | Povećanje → | Smanjenje → |
|-----------|----------|-------------|-------------|
| `POPULATION_SIZE` | 1000 | više raznovrsnosti, sporije | rizik preran konvergencije |
| `GENERATIONS` | 500 | bolji rezultat, sporije | premalo vremena za konvergenciju |
| `TOURNAMENT_SIZE` | 5 | jači pritisak, brža konvergencija | random drift |
| `CROSSOVER_RATE` | 0.85 | više mešanja | manje novih kombinacija |
| `MUTATION_RATE` | 0.15 | više exploration | prerana konvergencija |
| `ELITE_COUNT` | 10 | stabilnost, ali guši raznovrsnost | elita se može izgubiti |

---

**Kraj analize.**
