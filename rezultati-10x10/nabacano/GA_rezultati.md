# Rezultati genetskog algoritma za optimizaciju alokacije servisa

## 1 Parametri genetskog algoritma

| Parametar | Vrednost |
|---|---|
| Veličina populacije | 200 |
| Broj generacija | 1000 |
| Veličina turnira (selekcija) | 5 |
| Verovatnoća ukrštanja | 0.85 (85%) |
| Verovatnoća mutacije | 0.15 (15%) |
| Broj elitnih jedinki | 10 |

Hromozom je matrica dimenzija 10x10 celih brojeva, gde `chromosome[i][j]` predstavlja broj izvršavanja servisa `i` na računaru `j`.

---

## 2 Implementacija osnovnih koraka

### 2.1 Inicijalizacija populacije

Populacija od 200 jedinki se kreira kombinovano:

- **30% pohlepnih jedinki**: parovi (servis, računar) se sortiraju po odnosu profit/vreme od najboljeg ka najgorem, pa se pohlepno dodeljuje maksimalan moguć broj izvršavanja uz slučajno umanjenje od 0-33% radi diverzifikacije
- **70% slucajnih jedinki**: nasumične vrednosti unutar dozvoljenih granica

Svaka jedinka prolazi kroz repair funkciju koja garantuje izvodljivost.

### 2.2 Repair funkcija

Repair funkcija se poziva nakon svake genetičke operacije i obezbeđuje da je svako rešenje uvek validno. Radi u 4 koraka:

1. **Ispravljanje negativnih vrednosti na 0**
2. **Ograničenje tražnje po servisu** — ako ukupno izvršavanja servisa prelazi dozvoljenu tražnju, proporcionalno se smanjuju sve vrednosti u tom redu
3. **Ograničenje vremena po računaru** — ako računar prelazi 2880 minuta, iterativno se uklanjaju jedinice servisa sa najlošijim odnosom profit/vreme dok se ne zadovolji ograničenje
4. **Popunjavanje preostalog vremena** — preostalo slobodno vreme na računaru se popunjava najprofitabilnijim servisima koji još imaju slobodnu tražnju

### 2.3 Selekcija (turnirska)

Koristi se turnirska selekcija sa veličinom turnira 5. Nasumično se bira 5 jedinki iz populacije, a jedinka sa najvećom zaradom se bira kao roditelj. Proces se ponavlja dva puta za izbor oba roditelja.

### 2.4 Ukrštanje (uniformno)

Sa verovatnoćom od 85% vrši se uniformno ukrštanje. Za svaku ćeliju matrice (servis, računar) sa verovatnoćom 50% vrednost se uzima od prvog roditelja, inače od drugog. Nastaju dva deteta koja prolaze kroz repair funkciju.

### 2.5 Mutacija (tri operatora)

Primenjuju se tri operatora mutacije:

1. **Tačka mutacija (1.5% po ćeliji)**: Za svaku ćeliju sa verovatnoćom 1.5%, vrednost se promeni za slučajni delta u opsegu ±33% trenutne vrednosti.
2. **Zamena kolona (5%)**: Sa verovatnoćom 5%, nasumično se biraju dva računara i kompletno se zamenjuju sve njihove alokacije servisa.
3. **Preraspoređivanje reda (5%)**: Sa verovatnoćom 5%, bira se jedan servis i ukupan broj izvršavanja se preraspodeljuje po računarima srazmerno odnosu profit/vreme svakog računara za taj servis.

### 2.6 Elitizam i detekcija stagnacije

Top 10 jedinki iz svake generacije direktno prelazi u sledeću bez izmena (elitizam). Ako nema poboljšanja 100 uzastopnih generacija, najlošijih 20% populacije (40 jedinki) se zamenjuje novim slučajnim jedinkama radi očuvanja diverzifikacije.

---

## 3 Rezultati genetskog algoritma

### 3.1 Konvergencija kroz generacije (kumulativni maksimum)

| Generacija | Najbolji fitness | Prosečni fitness | Kumulativni maksimum |
|---|---|---|---|
| 0 | 87,096 | 74,339.6 | 87,096 |
| 50 | 88,854 | 88,548.7 | 88,854 |
| 100 | 89,037 | 88,872.3 | 89,037 |
| 150 | 89,108 | 88,847.1 | 89,108 |
| 200 | 89,166 | 88,628.1 | 89,166 |
| 250 | 89,208 | 88,626.2 | 89,208 |
| 300 | 89,208 | 88,916.1 | 89,208 |
| 350 | 89,245 | 88,831.1 | 89,245 |
| 400 | 89,252 | 88,907.3 | 89,252 |
| 450 | 89,252 | 88,905.6 | 89,252 |
| 500 | 89,332 | 89,135.7 | 89,332 |
| 550 | 89,349 | 89,021.6 | 89,349 |
| 600 | 89,357 | 89,166.4 | 89,357 |
| 650 | 89,357 | 89,070.5 | 89,357 |
| 700 | 89,379 | 89,131.2 | 89,379 |
| 750 | 89,425 | 89,094.5 | 89,425 |
| 800 | 89,445 | 89,251.9 | 89,445 |
| 850 | 89,445 | 89,087.7 | 89,445 |
| 900 | 89,459 | 89,258.3 | 89,459 |
| 950 | 89,470 | 89,111.2 | 89,470 |

**Finalni rezultat: 89,474 dinara**

### 3.2 Matrica alokacije (GA resenje)

|  | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 | C9 | C10 |
|---|---|---|---|---|---|---|---|---|---|---|
| **S1** | 0 | 0 | 0 | 0 | 0 | 0 | 126 | 0 | 0 | 0 |
| **S2** | 480 | 0 | 0 | 0 | 0 | 120 | 0 | 0 | 0 | 0 |
| **S3** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **S4** | 0 | 295 | 0 | 0 | 0 | 0 | 505 | 0 | 0 | 0 |
| **S5** | 0 | 138 | 0 | 166 | 0 | 71 | 23 | 1 | 0 | 1 |
| **S6** | 0 | 0 | 576 | 0 | 552 | 0 | 0 | 0 | 71 | 0 |
| **S7** | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 295 | 0 |
| **S8** | 0 | 0 | 0 | 4 | 17 | 190 | 0 | 410 | 55 | 24 |
| **S9** | 0 | 0 | 0 | 149 | 0 | 0 | 0 | 0 | 51 | 0 |
| **S10** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 450 |

### 3.3 Iskorišćenje računara (GA)

| Računar | Korišćeno (min) | Kapacitet (min) | Iskorišćenje |
|---|---|---|---|
| C1 | 2880 | 2880 | 100.0% |
| C2 | 2879 | 2880 | 100.0% |
| C3 | 2880 | 2880 | 100.0% |
| C4 | 2880 | 2880 | 100.0% |
| C5 | 2879 | 2880 | 100.0% |
| C6 | 2880 | 2880 | 100.0% |
| C7 | 2880 | 2880 | 100.0% |
| C8 | 2880 | 2880 | 100.0% |
| C9 | 2879 | 2880 | 100.0% |
| C10 | 2878 | 2880 | 99.9% |

### 3.4 Iskorišćenje tražnje servisa (GA)

| Servis | Alocirano | Tražnja | Iskorišćenje |
|---|---|---|---|
| S1 | 126 | 1000 | 12.6% |
| S2 | 600 | 600 | 100.0% |
| S3 | 0 | 500 | 0.0% |
| S4 | 800 | 800 | 100.0% |
| S5 | 400 | 400 | 100.0% |
| S6 | 1199 | 1200 | 99.9% |
| S7 | 300 | 300 | 100.0% |
| S8 | 700 | 700 | 100.0% |
| S9 | 200 | 200 | 100.0% |
| S10 | 450 | 450 | 100.0% |

---

## 4 Rezultati NEOS servera (celobrojno linearno programiranje)

### 4.1 Informacije o solveru

| Parametar | Vrednost |
|---|---|
| NEOS Job ID | 18666557 |
| Solver | SCIP 10.0.0 |
| Metoda | MILP (Mixed Integer Linear Programming) |
| LP solver | SoPlex 8.0.0 |
| Vreme rešavanja | 0.19 sekundi |
| Broj promenljivih | 100 (celobrojne) |
| Broj ograničenja | 20 |
| Broj čvorova (B&B stablo) | 25 |
| Broj pronađenih rešenja | 178 |
| Gap (primal-dual) | 0.00% |

### 4.2 Matrica alokacije (NEOS rešenje)

|  | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 | C9 | C10 |
|---|---|---|---|---|---|---|---|---|---|---|
| **S1** | 0 | 0 | 0 | 0 | 0 | 1 | 160 | 0 | 0 | 0 |
| **S2** | 480 | 0 | 0 | 0 | 0 | 120 | 0 | 0 | 0 | 0 |
| **S3** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **S4** | 0 | 280 | 0 | 0 | 0 | 0 | 520 | 0 | 0 | 0 |
| **S5** | 0 | 148 | 0 | 128 | 0 | 122 | 0 | 1 | 0 | 1 |
| **S6** | 0 | 0 | 576 | 0 | 576 | 0 | 0 | 0 | 6 | 0 |
| **S7** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 300 | 0 |
| **S8** | 0 | 0 | 0 | 0 | 0 | 116 | 0 | 410 | 150 | 24 |
| **S9** | 0 | 0 | 0 | 200 | 0 | 0 | 0 | 0 | 0 | 0 |
| **S10** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 450 |

**Optimalna zarada: 89,847 dinara**

---

## 5 Uporedna analiza

### 5.1 Poređenje rezultata

| Metrika | Genetski algoritam | NEOS (SCIP) |
|---|---|---|
| Maksimalna zarada | 89,474 din | 89,847 din |
| Razlika od optimuma | 373 din (0.42%) | 0 din (0.00%) |
| Vreme izvršavanja | ~60-120 sec | 0.19 sec |

### 5.2 Analiza konvergencije kumulativnog maksimuma

Genetski algoritam konvergira u karakterističnim fazama:

**Faza 1 — Brza konvergencija (generacije 0-50):** Fitness skače sa 87,096 na 88,854 (+1,758 din, +2.02%). Ovo je najznačajniji skok jer pohlepna inicijalizacija i repair funkcija brzo pronalaze dobre početne alokacije.

**Faza 2 — Umereno poboljšanje (generacije 50-500):** Fitness raste sa 88,854 na 89,332 (+478 din, +0.54%). Genetski operatori istražuju prostor rešenja i postepeno pronalaze bolje kombinacije.

**Faza 3 — Fino podešavanje (generacije 500-1000):** Fitness raste sa 89,332 na 89,474 (+142 din, +0.16%). Poboljšanja su sve manja jer se algoritam približava lokalnom optimumu.

Ukupno poboljšanje od inicijalne do finalne generacije: 89,474 - 87,096 = **2,378 din (+2.73%)**
