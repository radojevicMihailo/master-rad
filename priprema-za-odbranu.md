# Priprema za odbranu — detaljna razrada i odgovori na pitanja

Prateći dokument uz `prezentacija.pptx` i `tekst-za-izlaganje.md`.
Deo I razrađuje ono što stoji iza svakog slajda. Deo II sadrži pitanja koja komisija realno može da postavi, uključujući i ona iz samog rada koja nisu na slajdovima.

Sve brojke su iz `rad/main.pdf`. Referentni optimumi: **10×10 → 89.847**, **100×100 → 2.023.188**, **1000×1000 → 20.307.561** dinara. Radno vreme računara u svim eksperimentima: **T = 2880 minuta** (dva dana).

---

# DEO I — Razrada po slajdovima

## Slajd 1 — Naslov

Rad rešava problem maksimizacije zarade pri raspodeli servisa na skup računara različitih performansi. Dva pristupa se porede: egzaktni (celobrojno linearno programiranje, alat HiGHS) i stohastički (genetički algoritam sa specijalizovanom procedurom popravke). Kontrolna metoda je slučajna pretraga.

**Ako te pitaju „o čemu je rad u jednoj rečenici":** Formulisao sam problem raspodele poslova kao celobrojni linearni program, implementirao genetički algoritam sa namenskom procedurom popravke i eksperimentalno izmerio gde je koji pristup bolji na tri veličine instance.

## Slajd 2 — Model

Skupovi: $S = \{1,\dots,m\}$ servisi, $C = \{1,\dots,n\}$ računari.

| Oznaka | Značenje |
|---|---|
| $t_{ij}$ | vreme izvršavanja jedne jedinice servisa $i$ na računaru $j$ |
| $p_{ij}$ | zarada od te jedinice |
| $d_i$ | broj zahtevanih jedinica servisa $i$ |
| $T$ | maksimalno radno vreme svakog računara (2880 min) |
| $x_{ij}$ | **promenljiva odluke** — broj jedinica servisa $i$ na računaru $j$ |

$$\max \sum_i \sum_j p_{ij} x_{ij}$$
$$\sum_i t_{ij} x_{ij} \le T \quad \forall j \qquad \sum_j x_{ij} \le d_i \quad \forall i \qquad x_{ij} \in \mathbb{Z}_{\ge 0}$$

Broj promenljivih je $m \cdot n$, broj ograničenja $m + n$. Za 1000×1000: milion promenljivih, 2000 ograničenja.

**Zašto su računari „različitih performansi":** isti servis ne traje isto na svakoj mašini i ne donosi istu zaradu, jer mašine imaju različitu specijalizaciju. Zato $t_{ij}$ i $p_{ij}$ zavise od oba indeksa, a ne samo od servisa.

**Gornje granice promenljivih** unapred su sužene na $\min(d_i, \lfloor T/t_{ij}\rfloor)$ — to smanjuje linearnu relaksaciju i ubrzava grananje i ograničavanje.

## Slajd 3 — Šta je u radu standardno, a šta specifično

**Standardno (udžbenički GA):** turnirska selekcija, elitizam, jednotačkasto ukrštanje, mutacija.

**Specifično za ovaj problem:**
1. **Reprezentacija** — hromozom je matrica $X \in \mathbb{Z}_{\ge 0}^{m \times n}$. Genotip i fenotip su identični, nema kodiranja ni dekodiranja.
2. **Četvorofazna procedura popravke** — poziva se posle svake izmene hromozoma.

Cena direktne reprezentacije: skoro svaka izmena naruši dopustivost, pa je popravka obavezna. Korist: nema funkcija kazne (*penalty*), pa nema ni njihovog podešavanja.

## Slajd 4 — Procedura popravke

Četiri determinističke faze, **redosled je bitan**:

1. **Odsecanje negativnih vrednosti** → sve $< 0$ postaje 0. Mora prvo, jer bi negativne vrednosti pokvarile sume po redovima i kolonama.
2. **Ograničenje zahteva (po redovima)** → ako $\sum_j x_{ij} > d_i$, ceo red se proporcionalno smanjuje faktorom $d_i / \sum_j x_{ij}$. Čuva relativnu raspodelu po računarima.
3. **Ograničenje vremena (po kolonama)** → ako kolona prelazi $T$, uklanjaju se jedinice sa najnižim odnosom $p_{ij}/t_{ij}$ dok se ograničenje ne ispuni.
4. **Pohlepno popunjavanje** → dok ima slobodnog vremena i dok zahtev nije iscrpljen, dodaju se jedinice po opadajućem $p_{ij}/t_{ij}$.

Zašto taj redosled: korak 2 pre koraka 3 jer smanjenje po redovima oslobađa vreme i smanjuje posao u koraku 3. Korak 4 je poslednji jer pretpostavlja da su sva ograničenja već zadovoljena.

**Ključna optimizacija:** sortiranje servisa po $p_{ij}/t_{ij}$ radi se **jednom pre evolucije**, jer su $t_{ij}$ i $p_{ij}$ konstantni. Time trošak sortiranja izlazi iz unutrašnje petlje.

Koraci 3 i 4 su pohlepne lokalne heuristike — svako dete izlazi iz popravke istovremeno **dopustivo i lokalno poboljšano**.

## Slajd 5 — Postavka eksperimenta

| | 10×10 | 100×100 | 1000×1000 |
|---|---|---|---|
| Promenljivih | 100 | 10.000 | 1.000.000 |
| Ograničenja | 20 | 200 | 2.000 |
| Uloga | verifikaciona | referentna | granica resursa |
| $t_{ij}$ [min] | 4–17 | 5–24 | 5–24 |
| $p_{ij}$ [din] | 6–42 | 10–49 | 10–49 |
| $d_i$ | 200–1200 | 500 | 500 |

Sve instance su **vremenski zasićene**: na optimalnom rešenju iskorišćenost radnog vremena premašuje 99,9%.

Hardver: MacBook Pro 16" (2021), Apple M1 Pro, 16 GB RAM. g++ 13 sa `-O3 -mcpu=apple-m1`, Python 3.11, NumPy 1.26, HiGHS 1.7, Gurobi 11 preko NEOS-a.

Po jedna slučajno generisana instanca po veličini. Više nezavisnih pokretanja po konfiguraciji (15 na maloj, 2–11 na srednjoj, 3–6 na velikoj).

## Slajd 6 — GA naspram HiGHS

**Definicija „procenta optimuma":** odnos vrednosti funkcije cilja koju algoritam postiže i optimuma koji je dokazao HiGHS, izražen u procentima. **Na slajdovima 6, 7 i 9 to je uvek prosek svih pokretanja konfiguracije**, jer se uz njega prikazuje i prosečno vreme; najbolja pojedinačna pokretanja navedena su samo tamo gde je izričito rečeno.

| Instanca | HiGHS vrednost | HiGHS vreme | GA konfiguracija (P, G) | GA prosek | GA % (prosek) | GA najbolji % | GA vreme |
|---|---|---|---|---|---|---|---|
| 10×10 | 89.847 | 0,09 s | (1.000, 1.000) | 88.657 | 98,7 | 99,0 | 0,3 s |
| 100×100 | 2.023.188 | 7,7 s | (200.000, 200) | 1.999.503 | 98,8 | 98,8 | 1.779 s |
| 1000×1000 | 20.307.561 | 185 s | (20.000, 100) | 19.718.844 | 97,1 | 97,1 | 15.594 s |

Na maloj instanci konfiguracija (2.000, 500), takođe sa $10^6$ evaluacija, daje prosek 88.817 = **98,9%**, a najbolje pokretanje 89.010 = **99,1%**.

HiGHS na velikoj instanci: 184,7 s pri vremenskom limitu 150 s. Ista instanca je ranije rešena i za 465,6 s i za 1.059,6 s pri limitima 300 s i 1.000 s — solver menja strategiju prema zadatom limitu, ali svaki put dokaže isti optimum 20.307.561.

**Zaključak koji moraš da izgovoriš jasno:** na sve tri ispitane linearne instance HiGHS je bio i bolji i brži. GA nije zamena za egzaktni alat na ovom modelu; on je dopuna za slučajeve gde ILP formulacija nije primenljiva.

## Slajd 7 — GA naspram slučajne pretrage

Ovaj slajd koristi **konfiguracije usklađene po broju evaluacija** sa slučajnom pretragom, i za obe metode prikazuje **prosek pokretanja**, zato se vrednosti razlikuju od slajda 6.

| Instanca | Evaluacija (obe metode) | SP prosek % | GA konfiguracija (P, G) | GA prosek % |
|---|---|---|---|---|
| 10×10 | $10^5$ | 82,9 | (1.000, 100) | 98,3 |
| 100×100 | $10^7$ | 49,6 | (50.000, 200) | 98,5 |
| 1000×1000 | $2 \cdot 10^6$ | 88,0 | (20.000, 100) | 97,1 |

**Kritična tačka koju moraš da naglasiš:** slučajna pretraga koristi **istu** proceduru popravke kao GA. Svako slučajno generisano rešenje prolazi kroz iste četiri faze. Zato izmerena razlika ne meri „GA protiv slučajnosti", nego **isključivo doprinos evolutivnih operatora** — selekcije, ukrštanja, mutacije i elitizma.

**Zašto je 100×100 poseban:** tamo popravka slučajnog rešenja daje samo ~49% optimuma, pa evolucija nosi glavninu posla (skoro 50%). Na 10×10 i 1000×1000 popravka sama daje ~83% odnosno ~88%, pa evoluciji ostaje manje prostora. Objašnjenje je u Delu II, pitanje 5.2.

## Slajd 8 — Veličina populacije i konvergencija

Poređenje pri **istom ukupnom broju od $10^6$ evaluacija** na maloj instanci, 15 pokretanja po konfiguraciji:

| Konfiguracija | Rani napredak | Konačni prosek | Najgori slučaj |
|---|---|---|---|
| P = 1000, G = 1000, E = 30 | brži | 88.657 | 88.338 |
| P = 2000, G = 500, E = 50 | sporiji | **88.817** | **88.698** |

Manja populacija brže stiže do svog maksimuma (više generacija = više ciklusa selekcije nad istim materijalom). Veća populacija podiže **konačno dostignuti** maksimum, jer veća raznolikost smanjuje rizik od prerane konvergencije.

Isti obrazac na srednjoj instanci pri $10^7$ evaluacija: P = 50.000 vodi najveći deo pretrage, ali ga P = 100.000 prestiže između $5\cdot10^6$ i $10^7$ evaluacija (1.996.497 naspram 1.993.336 u proseku).

## Slajd 9 — Opadajući prinos

Instanca 100×100, rast računskog budžeta (x-osa grafikona je **broj evaluacija**, sa veličinom populacije u drugom redu oznake):

| P | G | Evaluacije | Prosek | % opt. | Vreme [s] | STDEV |
|---|---|---|---|---|---|---|
| 5.000 | 100 | 0,5 M | 1.981.686 | 97,95 | 16 | 2.534 |
| 10.000 | 100 | 1 M | 1.985.559 | 98,14 | 30 | 2.258 |
| 50.000 | 200 | 10 M | 1.993.336 | 98,52 | 311 | 757 |
| 100.000 | 100 | 10 M | 1.996.497 | 98,68 | 504 | 540 |
| 150.000 | 200 | 30 M | 1.997.704 | 98,74 | 1.383 | 213 |
| 200.000 | 200 | 40 M | 1.999.503 | 98,83 | 1.779 | 204 |

Sve vrednosti su **proseci pokretanja** (5–11 po konfiguraciji); referentni optimum je 2.023.188 dinara.

**10 M → 40 M evaluacija (P = 100.000 × G = 100 → P = 200.000 × G = 200): kvalitet +0,15% (98,68 → 98,83), vreme 504 s → 1.779 s (3,5×).**

Zasićenje počinje već oko P = 50.000 (98,52%). Standardna devijacija istovremeno pada sa 2.534 na 204 dinara — veća populacija daje i bolji i stabilniji rezultat, ali cena raste brže od koristi.

**Ako pitaju „zašto duplo veća populacija traje 3,5× duže":** nije duplirana samo populacija — u toj konfiguraciji je udvostručen i broj generacija, pa je ukupan posao 4× veći (10 M → 40 M evaluacija). Vreme dakle raste sporije od posla, ne brže.

Cena po evaluaciji:

| P × G | Evaluacije | Vreme [s] | µs/eval |
|---|---|---|---|
| 5.000 × 100 | 0,5 M | 16,3 | 32,6 |
| 10.000 × 100 | 1 M | 30,0 | 30,0 |
| 50.000 × 200 | 10 M | 311,4 | 31,1 |
| 100.000 × 100 | 10 M | 503,8 | 50,4 |
| 150.000 × 200 | 30 M | 1.382,6 | 46,1 |
| 200.000 × 200 | 40 M | 1.778,8 | 44,5 |

Vreme je približno linearno po broju evaluacija. Jedini efekat same veličine populacije je skok cene po evaluaciji sa ~31 na ~45–50 µs iznad P = 100.000 — veći memorijski otisak i slabija lokalnost pristupa (populacija ne staje u keš).

Isto se vidi i na paru sa istih 10 M evaluacija: P = 50.000 × 200 daje 98,52% za 311 s, a P = 100.000 × 100 daje 98,68% za 504 s. Pri istom budžetu evaluacija veća populacija daje bolji rezultat, ali je skuplja po evaluaciji.

## Slajd 10 — Ograničenja i domet zaključaka

Eksperimentalni zaključak važi za **ispitane statičke linearne instance**. Nije ispitano: nelinearnosti, dinamičke promene ulaza, više ciljeva, ograničenja koja se teško zapisuju linearno.

**Ako pitaju zašto se brojevi za veliku instancu razlikuju od slajda 6:** grafikon konvergencije prikazuje konfiguraciju (15.000, 100) sa 6 pokretanja i prosečnim vremenom 12.325 s = 3,4 h (prosek 96,97% optimuma). Najbolji rezultat na toj instanci (97,1%) daje konfiguracija (20.000, 100) sa 3 pokretanja i 15.594 s = 4,3 h po pokretanju, i taj broj stoji na slajdovima 6 i 7.

Šest pokretanja velike instance daje gotovo poklopljene krive — algoritam je stabilan (standardna devijacija ~4600 dinara, oko 0,02% optimuma). Ali jedno pokretanje traje 3,4 sata, što je i razlog zašto ih nije više.

**Najozbiljnije ograničenje evaluacije:** korišćena je po jedna instanca po veličini. Više pokretanja daje statističku pouzdanost u odnosu na stohastičnost algoritma, ali ne i u odnosu na varijabilnost instanci.

## Slajd 11 — Tri nalaza

1. GA daje stabilna rešenja kvaliteta **97–99%** referentnog optimuma.
2. GA je bolji od slučajne pretrage na sve tri instance, a razlika najizraženija na 100×100 (98,6% naspram 49,6%).
3. Veća populacija diže vrednost funkcije cilja, ali uz sve manji dobitak i brzo rastuće vreme.

## Slajd 12 — Zaključak

Glavni doprinos: **specijalizovana procedura popravke** koja posle svake izmene vraća rešenje u dopustivo stanje i lokalno ga poboljšava. HiGHS ostaje prvi izbor za ispitane linearne instance. Pravci daljeg rada: adaptivni operatori, lokalna pretraga, inicijalizacija iz LP relaksacije, dinamičke i višekriterijumske formulacije.

---

# DEO II — Pitanja i odgovori

## 1. Model i formulacija

### 1.1 Zašto ste ovaj problem formulisali baš kao ILP, a ne drugačije?
Zarada i vreme su linearni po $x_{ij}$ kad su $p_{ij}$, $t_{ij}$ i $d_i$ unapred poznati, a odluke su prirodno celobrojne — ne može se izvršiti pola jedinice servisa. To je tačno definicija celobrojnog linearnog programa. Linearni model je i prirodna osnova za kasnija proširenja: rešenje linearnog problema može poslužiti kao početna tačka za iterativne nelinearne metode.

### 1.2 Kojoj klasi problema ovo pripada?
Generalizacija problema ranca. Već za jedan računar svodi se na ograničeni problem ranca sa više predmeta, što je NP-težak problem. U opštem slučaju to je varijanta **generalizovanog problema dodeljivanja**, sa celobrojnim količinama umesto binarnih dodela.

### 1.3 Koliki je prostor pretrage?
Ako se zanemare ograničenja, $\prod_{i,j}\left(1 + \min(d_i, \lfloor T/t_{ij}\rfloor)\right)$. Za 1000×1000 to je reda $10^{2000}$ kandidatskih konfiguracija. Potpuna pretraga je neostvariva.

### 1.4 Funkcija cilja je linearna — zar onda nema lokalnih ekstrema? Čemu onda metaheuristika?
Tačno, u smislu diferencijalne analize nema klasičnih lokalnih ekstrema. Ali zbog celobrojnosti prostor je **diskretan**: susedna rešenja imaju različite vrednosti funkcije cilja, a promena jednog gena često zahteva simultanu promenu više drugih da bi rešenje ostalo dopustivo. Teškoća nije u obliku funkcije cilja, nego u kombinatornoj strukturi dopustivog skupa.

### 1.5 Da li ILP garantuje optimum?
Za **kontinualni** LP postoji matematički dokaz optimalnosti. Za **celobrojni** LP takav dokaz u opštem slučaju ne postoji u polinomskom vremenu — zato je ILP NP-težak. Ono što alat radi jeste da grananjem i ograničavanjem dokaže optimalnost za konkretnu instancu, i to je uspeo na sve tri moje instance. Formulacija „garantovani optimum" u opštem slučaju nije tačna, i u radu je preformulisana.

### 1.6 Zašto ograničenje zahteva ide sa $\le$, a ne sa $=$?
Zato što $d_i$ predstavlja tržišni zahtev — gornju granicu onoga što tržište može da apsorbuje. Nije obavezno realizovati sve. Ako neki servis nije profitabilan ni na jednom računaru u odnosu na alternative, optimalno rešenje ga jednostavno ne izvršava. Sa jednakošću bi problem mogao da postane i nedopustiv.

### 1.7 Gde ovo ima primenu?
Cloud računarstvo (servisi = zahtevi korisnika, računari = virtuelne mašine, zarada = prihod provajdera), industrijska proizvodnja (artikli i proizvodne linije), logistika (tipovi paketa i distributivni centri). Matematička struktura je ista, menja se samo interpretacija.

## 2. Genetički algoritam — dizajn

### 2.1 Zašto ste izabrali baš genetički algoritam?
Tri razloga. Prvo, prostor pretrage je diskretan i visokodimenzionalan, što odgovara osnovnoj postavci GA. Drugo, rešenja koja narušavaju ograničenja mogu se procedurom popravke efikasno vratiti u dopustivu oblast, čime se izbegavaju funkcije kazne koje komplikuju podešavanje. Treće, funkcija cilja je separabilna po članovima, pa se uticaj pojedinačnog gena lako analizira.

### 2.2 Zašto je matrica dobra reprezentacija?
Nije da je „prirodnija zato što je celobrojna" — to bi bila prazna tvrdnja. Konkretno: promenljive modela su $x_{ij}$, pa matrica $m \times n$ **jeste** rešenje, bez ijednog koraka kodiranja ili dekodiranja. Genotip i fenotip su identični. To znači da je funkcija cilja direktno računljiva nad hromozomom i da nema gubitka informacije u prevođenju. Cena je obavezna kontrola dopustivosti posle svake izmene.

### 2.3 Koji su operatori i kako rade?
- **Selekcija:** turnirska, veličina turnira $k$. Nasumično se izabere $k$ hromozoma sa ponavljanjem, pobeđuje onaj sa najvećom vrednošću funkcije cilja. Ne zahteva normalizaciju vrednosti, a selekcioni pritisak se kontroliše sa $k$.
- **Elitizam:** najboljih $E$ hromozoma prelazi u sledeću generaciju neizmenjeno. Garantuje da najbolja vrednost monotono ne opada.
- **Ukrštanje:** jednotačkasto, $p_c = 0{,}8$. Matrice se ravnaju u niz dužine $mn$ po redovima, bira se tačka preseka, deca razmenjuju prefiks i sufiks.
- **Mutacija:** $p_m = 0{,}15$. Bira se $L$ slučajnih gena i svaki se **zamenjuje** slučajnim brojem iz $[0, \lfloor T/t_{ij}\rfloor]$ — nije mala perturbacija, nego potpuna zamena, što omogućava velike skokove.

### 2.4 Zašto se mutira 50 gena u C++ verziji, a jedan u Python verziji?
Zato što se broj mutiranih gena mora skalirati sa dimenzijom. Promena jednog gena od $10^6$ na velikoj instanci ima potpuno zanemarljiv efekat na vrednost funkcije cilja. U referentnoj Python implementaciji, na maloj instanci sa 100 gena, jedan gen je 1% hromozoma i to je dovoljno.

### 2.5 Nije li $p_m = 0{,}15$ visoko za genetički algoritam?
Jeste, u odnosu na uobičajene preporuke. Podnošljivo je zato što procedura popravke deluje kao stabilizator: ona odmah otkloni destruktivne efekte mutacije i još lokalno poboljša rešenje, a elitne jedinke su u svakom slučaju zaštićene. Sistematsko ispitivanje na maloj instanci to i potvrđuje — kvalitet monotono raste sa stopom mutacije i ulazi u plato oko $p_m = 0{,}4$–$0{,}6$, **bez degradacije čak ni pri $p_m = 1{,}0$**. Vrednost 0,15 je konzervativan izbor.

| $p_m$ | 0,05 | 0,10 | 0,15 | 0,25 | 0,40 | 0,60 | 0,80 | 1,00 |
|---|---|---|---|---|---|---|---|---|
| % opt. (prosek) | 97,9 | 98,2 | 98,3 | 98,4 | 98,5 | 98,6 | 98,6 | 98,6 |

Ograničenje ovog zaključka: važi za mutaciju **jednog** gena. Kod C++ konfiguracija gde se menja grupa od 50 gena, destruktivni efekat po događaju je veći, pa je umerena stopa opravdanija.

### 2.6 Šta je injekcija raznolikosti i da li je pomogla?
Ako se najbolja vrednost ne poboljša 100 uzastopnih generacija, 20% najlošijih hromozoma zamenjuje se novim slučajnim jedinkama.

**Iskren odgovor: u mojim eksperimentima nije demonstriran njen doprinos.** Uslov se aktivira samo u dugim pokretanjima na maloj instanci — u svih 15 pokretanja konfiguracije (1000, 1000), u 14 od 15 za (2000, 500), u 12 od 15 za (100, 1000). Na srednjoj i velikoj instanci nikad, jer se najbolja vrednost pomera skoro svake generacije. A tamo gde jeste okinula, nije zabeleženo naknadno poboljšanje: ubačene jedinke startuju oko 80% optimuma naspram elite na oko 99%, pa kroz turnirsku selekciju retko dođu do reprodukcije. Mehanizam je zadržan kao osiguranje, ali bi za tvrdnju o njegovoj korisnosti trebala ablaciona studija.

### 2.7 Zašto niste koristili funkcije kazne umesto popravke?
Kazne zahtevaju podešavanje težinskih koeficijenata, koje je osetljivo i po instanci različito. Pored toga, sa kaznama populacija sadrži nedopustiva rešenja, pa se deo evaluacija troši na region koji nas ne zanima. Popravka garantuje da je **svaki** hromozom u populaciji dopustiv, pa dodatni članovi nisu potrebni i podešavanje je jednostavnije.

### 2.8 Kada se algoritam zaustavlja?
Posle unapred zadatog broja generacija $G$. Izabrano zato što daje unapred poznato vreme izvršavanja i olakšava pošteno poređenje konfiguracija. Alternative iz literature — zaustavljanje po isteku vremena, po stagnaciji, po padu raznovrsnosti ispod praga — nisu korišćene, ali se prati kriva kumulativnih maksimuma, pa se retrospektivno vidi da li je $G$ bio dovoljan.

## 3. Hiperparametri

### 3.1 Kako ste ih podešavali?
Ne iscrpnim pretraživanjem, nego unapred uparenim konfiguracijama, u dva režima: pri **fiksnom broju evaluacija** ($P \cdot G$ konstantno) i pri **fiksnom broju generacija**. Prvi režim meri kako najbolje potrošiti dati budžet, drugi meri marginalnu korist dodatnih resursa.

### 3.2 Šta je bolje — veća populacija ili više generacija?
**Zavisi od budžeta evaluacija, i to je jedan od zanimljivijih nalaza.**

Pri skromnom budžetu ($10^5$ evaluacija, mala instanca) manja populacija sa više generacija je bolja: (200, 500) daje 98,7%, a (2000, 50) samo 98,2%. Više generacija znači više ciklusa selekcije i rekombinacije nad istim genetičkim materijalom.

Pri velikom budžetu poredak se **obrće**: na maloj instanci pri $10^6$ i na srednjoj pri $10^7$ evaluacija veća populacija dostiže bolju konačnu vrednost, jer manja populacija ranije iscrpi raznolikost pa dodatne generacije prestanu da donose napredak.

### 3.3 Zašto veličina turnira mora da raste sa populacijom?
Selekcioni pritisak zavisi od **odnosa** $k$ i veličine populacije, ne od $k$ samog. Fiksni turnir $k = 5$ u populaciji od 20.000 znači da pobednik dolazi iz uzorka od 0,025% populacije — pritisak je premali. Za $k = 1$ selekcija je potpuno slučajna; za $k = P$ uvek pobeđuje globalno najbolji i raznolikost se gubi odmah.

**Konkretan dokaz da to košta:** na velikoj instanci konfiguracija (10000, 100) sa $k = 10$ daje **lošiji** prosek (96,6%) od manje konfiguracije (8000, 100) sa $k = 15$ (96,9%). To je jedino odstupanje od monotonog trenda po populaciji i objašnjava se upravo turnirom.

### 3.4 Koliko elitizma?
3–5% populacije. Na maloj instanci oko 3%, na srednjoj 1–5%, na velikoj ispod 1% — ali i to je već oko stotinu jedinki. Previše elite ubrzava konvergenciju i povećava rizik od zaglavljivanja.

### 3.5 Koliko je algoritam osetljiv na hiperparametre?
Relativno neosetljiv na umerene promene, na svim instancama. To je poželjno u praksi jer smanjuje potrebu za skupim podešavanjem po svakoj novoj instanci. Robusnost je delom posledica snage ugrađenih lokalnih heuristika, a delom stabilizujućeg dejstva same populacione pretrage.

## 4. Rezultati i poređenje sa HiGHS

### 4.1 Ako je HiGHS bolji i brži na sve tri instance, čemu onda ceo genetički algoritam?
**Ovo je najverovatnije pitanje na odbrani. Odgovori direktno, bez izvrdavanja.**

Rad je merenje, a ne reklama za metaheuristiku. Nalaz da HiGHS pobeđuje na sve tri ispitane instance je legitiman rezultat i eksplicitno je tako i napisan: GA je **dopuna, a ne zamena**.

Vrednost GA leži van ispitanog režima, i to u tri konkretne situacije:
1. **Kada model prestane da bude linearan.** Ako zarada po jedinici zavisi od ukupno izvršenog broja jedinica (količinski popusti, zasićenje tržišta) ili ako se vreme izvršavanja menja sa opterećenjem mašine, ILP formulacija više ne važi. GA ne mari — funkcija cilja mu je crna kutija.
2. **Kada memorijski zahtevi ILP modela premaše hardver.** ILP mora da drži ceo model i drvo pretrage; GA drži populaciju i može se skalirati naniže.
3. **Kod višekriterijumskih formulacija**, gde se traži Pareto front, a ne jedna optimalna tačka.

Ono što rad **dokazuje** jeste gde je granica: za statične linearne instance do reda milion promenljivih, HiGHS je pravi izbor. To je koristan inženjerski nalaz.

### 4.2 A kako GA daje brz odziv ako je HiGHS brži? (pitanje iz komentara mentora)
Ne daje — na ispitanim instancama, i to treba reći otvoreno. Proverio sam to i pri **jednakom raspoloživom vremenu**: jedna generacija referentne konfiguracije na velikoj instanci traje oko 123 sekunde. Za 185 sekundi, koliko HiGHS-u treba da dokaže optimum, GA stigne da obradi jednu do dve generacije i nalazi se na oko 88–89% optimuma. Do svojih 97% dolazi tek posle više sati. Dakle GA nema prednost ni u režimu ograničenog vremena. Njegova prednost je u fleksibilnosti modela, ne u brzini.

### 4.3 Da li ste proverili da su HiGHS rezultati tačni?
Delimično — i to treba reći precizno. Nezavisna provera rađena je Gurobijem 11 preko NEOS servera. Potvrdio je istu vrednost na **10×10** i na **100×100** (u kraćem vremenu, nekoliko sekundi). Model velike dimenzije, sa milion promenljivih, **nije mogao da se reši preko NEOS-a**, pa za instancu 1000×1000 nezavisna potvrda ne postoji.

**Ako te pitaju je li onda referenca na velikoj instanci pouzdana:** HiGHS je pri uklonjenom vremenskom limitu *dokazao* optimalnost vrednosti 20.307.561 u oko 185 sekundi — dakle to nije samo najbolje pronađeno rešenje, nego dokazan optimum u okviru samog alata. Nezavisna potvrda drugim alatom bi bila dodatna sigurnost, i to je nedostatak koji priznajem, ali dokaz optimalnosti daje sam HiGHS. Poklapanje sa Gurobijem na obe manje instance dodatno govori u prilog pouzdanosti alata.

### 4.4 Zašto je HiGHS, a ne Gurobi ili CPLEX, primarni alat?
HiGHS je open-source i dostupan bez licencnih ograničenja, što je važno za ponovljivost rada. Komercijalni alati su brži na izuzetno velikim instancama, ali razlika u **vrednosti rešenja** u mojim eksperimentima nije postojala. Model je implementiran u Pythonu preko biblioteke PuLP, koja služi kao apstrakcioni sloj, pa je zamena alata trivijalna.

### 4.5 Koliko su rezultati stabilni?
Vrlo. Na maloj instanci interkvartilni opseg preko 15 pokretanja je ispod 200 dinara, standardna greška proseka oko 38 dinara (< 0,05% optimuma). Na srednjoj standardna devijacija najviše 0,13% najbolje vrednosti. Na velikoj oko 4600 dinara, što je oko 0,02% optimuma.

Zanimljivo: **relativna** devijacija opada sa dimenzijom (0,15% → 0,1% → 0,02%), iako apsolutna raste. Sa rastom broja gena pojedinačne slučajne odluke se usrednjavaju, pa je algoritam na velikim instancama srazmerno stabilniji.

### 4.6 Zašto samo 6 pokretanja na velikoj instanci?
Jedno pokretanje traje 3,4 sata. Šest pokretanja je oko 20 sati čistog računanja na jednoj mašini. To je bio praktičan gornji limit. Krive tih šest pokretanja se grupišu u vrlo uskom opsegu, pa je zaključak o stabilnosti opravdan i sa tim brojem.

## 5. Slučajna pretraga i doprinos popravke

### 5.1 Kako slučajna pretraga dostiže čak 83% na maloj instanci? Zar to nije previše za nasumično pogađanje?
Zato što **nije** nasumično pogađanje. Svako slučajno generisano rešenje prolazi kroz istu proceduru popravke kao i hromozomi GA, uključujući i pohlepno popunjavanje u četvrtom koraku. Taj korak je informisana konstrukcija — dodaje najprofitabilnije jedinice po $p_{ij}/t_{ij}$. Dakle tih ~83% (prosek 82,9%, najbolje pokretanje 83,8%) je zasluga popravke, a ne pretrage. To sam eksplicitno naveo na mestu gde se slučajna pretraga prvi put pominje, da čitalac zna šta gleda u tabelama.

### 5.2 Zašto je slučajna pretraga na 100×100 pala na 49,6%, a na 1000×1000 se vratila na 88%? To nije monotono.
**Ovo je najsuptilniji nalaz u radu i vredi ga znati napamet.**

U sva tri slučaja popravka iskoristi preko 99,9% raspoloživog radnog vremena. Razlika, dakle, nije u neiskorišćenim resursima, nego u **profitabilnosti jedinica koje to vreme zauzimaju**.

- **Na 100×100:** proporcionalno smanjivanje redova u drugom koraku popravke **očuva strukturu** slučajnog rešenja. Vreme računara ostaje zauzeto pretežno slabo profitabilnim dodelama, a pohlepno popunjavanje nema slobodnog vremena da ih ispravi. Otud samo 49%.
- **Na 1000×1000:** isti zahtev od 500 jedinica raspodeljuje se preko hiljadu računara, pa celobrojno odsecanje pri proporcionalnom smanjivanju **obriše veliku većinu gena** — posle popravke je svega oko 12% gena nenulto. Pohlepno popunjavanje tada gradi rešenje praktično iznova i dostiže kvalitet blizak čistoj pohlepnoj heuristici, oko 88%.

Zaključak: doprinos procedure popravke **ne može se uopštiti** — zavisi od odnosa dimenzija problema i strukture ograničenja.

### 5.3 Koliki je onda stvarni doprinos evolucije?
Meri se kao razlika između kvaliteta najbolje jedinke početne populacije i konačnog rezultata:

| Instanca | Početna populacija | Konačno | Doprinos evolucije |
|---|---|---|---|
| 10×10 | ~81% (72.673) | 98,3% | ~17% |
| 100×100 | ~49% (995.553) | 98,5% | **~50%** |
| 1000×1000 | ~88% (~17,85 M) | 97,1% | ~9% |

Na srednjoj instanci evolutivna komponenta nosi glavninu ukupnog kvaliteta. Tvrdnja da je „kvalitet uglavnom zasluga lokalnih heuristika" bila bi netačna kao opšta tvrdnja.

### 5.4 Zašto se kriva slučajne pretrage tako brzo zaravni?
Verovatnoća da nezavisno izvučen uzorak nadmaši dotadašnji maksimum opada sa brojem uzoraka, pa kumulativni maksimum raste tek logaritamski sporo. Konkretno na velikoj instanci: najbolja vrednost među prvih 20.000 uzoraka bila je 17.854.707, a dodatnih 1,98 miliona evaluacija donelo je poboljšanje od svega **0,05%**. Sa rastom dimenzije efekat je izraženiji, jer udeo visokokvalitetnih rešenja u prostoru pretrage eksponencijalno opada.

### 5.5 Je li poređenje pošteno kad GA i slučajna pretraga nemaju identičan broj evaluacija na velikoj instanci?
Na maloj i srednjoj instanci broj evaluacija je identičan ($10^5$, odnosno $10^7$). Na velikoj slučajna pretraga je imala **više** — 2 miliona naspram 1,5 miliona za GA — i uz to tri puta duže vreme (10,6 sati naspram 3,4). Dakle poređenje je u korist slučajne pretrage, a ona i dalje gubi sa 88,0% naspram 97,0%.

## 6. Implementacija i složenost

### 6.1 Zašto dve implementacije?
Python sa NumPy je referentna verzija — čitljiva, vektorizovana, oko 5 puta brža od naivne Python implementacije. Za velike instance je preskupa, pa je razvijena C++ verzija, koja je 20 do 50 puta brža. Za 1000×1000 to je razlika između nekoliko sati i nekoliko minuta po pokretanju.

### 6.2 Kolika je vremenska složenost?
Po generaciji:
- funkcija cilja: $O(P \cdot mn)$
- sortiranje za elitizam: $O(P \log P)$
- turnirska selekcija: $O(Pk)$
- ukrštanje i mutacija: $O(P \cdot mn)$
- popravka (najskuplji korak, popunjavanje): $O(mn \log m)$ po hromozomu

Ukupno po generaciji $O(P \cdot mn \log m)$, za ceo algoritam $O(G \cdot P \cdot mn \log m)$.

Provera brojkama: za 100×100 sa $P = 5000$, $G = 100$ to je reda $3 \cdot 10^{10}$ operacija → izmereno 16 s. Za 1000×1000 sa $P = 15000$, $G = 100$ reda $1{,}5 \cdot 10^{13}$ → izmereno 12.300 s. Odnos izmerenih vremena prati odnos procenjenih operacija po redu veličine.

### 6.3 Kako ste stali u 16 GB memorije sa populacijom od 15.000 hromozoma po milion gena?
Nominalno bi to bilo $1{,}5 \cdot 10^{10}$ vrednosti, odnosno oko **60 GB** pri standardnih 4 bajta. Tri stvari to obaraju:
1. **16-bitni zapis gena** — dovoljno jer su vrednosti ograničene sa $\min(d_i, \lfloor T/t_{ij}\rfloor)$. Prepolovljuje nominalnu zauzetost.
2. **Semantika premeštanja umesto kopiranja** pri zameni generacija; skupo kopiranje se radi samo za elitu.
3. **Retkost hromozoma** — posle popravke je svega oko 12% gena nenulto, pa kompresija memorije operativnog sistema efikasno sažima stranice ispunjene nulama.

Izmerena rezidentna zauzetost tokom izvršavanja: oko **4 GB**.

### 6.4 Zašto ste birali populaciju 15.000 na velikoj instanci?
Zato što i vreme i memorija rastu **linearno** sa $P$, pa postoji prirodna gornja granica koju hardver podržava. 15.000 sa 100 generacija je bio kompromis koji staje u 16 GB i u prihvatljivo vreme po pokretanju.

### 6.5 Zašto linearizovana matrica, a ne ugnežđene strukture?
Zbog lokalnosti u kešu. Izračunavanje funkcije cilja i popravka svode se na sekvencijalne prolaske kroz memoriju, koje dobro koriste keš i hardversko pretpreuzimanje. Ugnežđene strukture sa razbacanim alokacijama vodile bi čestim promašajima keša — dobro poznat efekat kod algoritama ograničenih propusnošću memorije. Na ovoj veličini niskonivouska optimizacija je jednako važna kao algoritamski izbori.

## 7. Konvergencija

### 7.1 Kako izgleda tipična kriva konvergencije?
Tri faze, iste na sve tri instance:
1. **Prvih 5–10% generacija:** izuzetno brz rast — selekcija eliminiše najlošije hromozome iz početne populacije.
2. **Narednih 20–30%:** umeren rast kroz rekombinaciju gradećih blokova iz najboljih hromozoma.
3. **Preostalih 60–70%:** spor ali stabilan rast kroz mutacije i lokalne popravke.

### 7.2 Šta pokazuje log-log skala?
Približno linearan trend u srednjem opsegu generacija, što znači da brzina konvergencije sledi polinomski zakon sa eksponentom manjim od jedinice. Eksponent **opada sa dimenzijom**: ~0,5 za 10×10, ~0,4 za 100×100, ~0,3 za 1000×1000. Time se kvantifikuje opšta zakonitost da relativna brzina konvergencije opada sa rastom dimenzije — jedno od osnovnih teorijskih ograničenja stohastičkih metaheuristika.

### 7.3 Zašto je kriva monotono neopadajuća?
Zbog same definicije kumulativnog maksimuma — u svakoj generaciji prikazuje najbolju **do tada** pronađenu vrednost. Elitizam dodatno garantuje da najbolja vrednost u samoj populaciji ne opada. Da nema elitizma, najbolje pronađeno rešenje bi se čuvalo sa strane, što je uobičajena praksa kod stohastičkih algoritama.

### 7.4 Da li je 100 generacija bilo dovoljno na velikoj instanci?
Kriva na kraju još uvek raste, dakle nije dostignut plato. Više generacija bi verovatno donelo dodatnih nekoliko desetina bazičnih poena, ali uz linearan rast vremena — a već 100 generacija traje 3,4 sata. Zaustavljanje po broju generacija izabrano je upravo zato što daje unapred poznato vreme i pošteno poređenje konfiguracija.

## 8. Ograničenja i kritika

### 8.1 Koje je najozbiljnije ograničenje vašeg rada?
**Po jedna instanca po veličini problema.** Više pokretanja daje statističku pouzdanost u odnosu na stohastičnost algoritma, ali ne i u odnosu na varijabilnost instanci. Analiza iz odeljka o početnoj populaciji upravo pokazuje da ponašanje osetno zavisi od strukture instance — doprinos popravke se kreće od 49% do 88% u zavisnosti od instance. Za čvršće uopštavanje bilo bi potrebno više instanci po veličini, sa različitim raspodelama vremena, zarada i zahteva. To je i deseti pravac daljeg rada.

### 8.2 Ostala ograničenja?
- Kvalitet zavisi od podešavanja hiperparametara; univerzalna konfiguracija ne postoji.
- Nema garancije optimalnosti — na velikoj instanci ostaje oko 3% razlike. U aplikacijama gde je optimalnost neophodna, GA nije zamena za ILP.
- Vreme za velike instance ostaje visoko: 3,4 sata po pokretanju.
- Algoritam je za **statičke** instance; dinamičke varijante nisu pokrivene.
- Pretpostavka o nezavisnosti servisa može biti narušena kod deljenih resursa.
- Model ne uključuje otkaze računara.
- Funkcija cilja je jednokriterijumska.

### 8.3 Da li se ovo može zvati memetskim algoritmom?
Ne bih ga tako zvao. Ovo je genetički algoritam sa procedurom popravke; sama popravka može se posmatrati i kao deo obrade dopustivosti, a ne kao zaseban lokalni pretraživač u memetskom smislu. Termin bi ovde više zamaglio nego pojasnio.

### 8.4 Da li ste zaista dokazali da GA vredi?
Dokazao sam nekoliko konkretnih stvari: da GA sa popravkom stabilno dostiže 97–99% optimuma, da je razlika u odnosu na slučajnu pretragu sa istim brojem evaluacija ogromna i da potiče isključivo od evolutivnih operatora, i da na ispitanim linearnim instancama HiGHS ostaje bolji izbor. Nisam dokazao da je GA bolji od ILP-a, i ne tvrdim to nigde u radu.

## 9. Dalji rad

### 9.1 Šta biste sledeće uradili?
Redosled po očekivanoj koristi:
1. **Inicijalizacija iz LP relaksacije** — rešenje relaksiranog linearnog programa kao početna populacija. Pretpostavka je da bi se preostala 3% na velikim instancama tako smanjila. Ovo je najkonkretniji i najizvodljiviji pravac.
2. **Eksplicitna lokalna pretraga** u evolutivnoj petlji, na primer razmene jedinica između računara po principu 2-opt.
3. **Adaptivne stope** mutacije i ukrštanja vezane za trenutnu raznolikost populacije, čime bi se izbeglo ručno podešavanje.
4. **Specijalizovani operatori ukrštanja** koji uvažavaju strukturu problema — razmena celih kolona (raspored po računaru) ili redova (raspored servisa preko računara), umesto slepog jednotačkastog preseka nad linearizovanim nizom.
5. **Više instanci po veličini**, radi provere koliko su izmereni odnosi osetljivi na strukturu instance.

Dalje, van neposrednog dometa: dinamičke instance sa inkrementalnom popravkom, mašinsko učenje za predikciju dobrih početnih rešenja, Lagranžova relaksacija za usmeravanje pretrage, i višekriterijumske formulacije (NSGA-II, SPEA2) za balansiranje zarade sa energetskom potrošnjom i ravnomernošću opterećenja.

### 9.2 Koja je vaša praktična preporuka nekome ko ovaj problem rešava?
Za male i srednje instance, do nekoliko hiljada promenljivih: direktno HiGHS. Za velike instance: i tamo HiGHS, sve dok memorijski zahtevi punog ILP modela ne premaše hardver ili dok se proširenja modela mogu izraziti linearno. Genetički algoritam postaje praktičan izbor tek kada jedan od ta dva uslova padne.

Ako se ipak koristi GA: $p_c = 0{,}8$ i $p_m = 0{,}15$ su robusni na svim veličinama. Ostalo skalirati — male instance: populacija ~1000, ~1000 generacija, turnir ~5, elita ~3%. Velike instance: populacija desetine hiljada, ~100 generacija, turnir 15–20, elita ispod 5%. Pri ograničenom broju evaluacija po pravilu je isplativije smanjiti populaciju u korist broja generacija.

---

# Brzi podsetnik pred ulazak

**Brojevi koje moraš znati napamet:**
- Optimumi: 89.847 / 2.023.188 / 20.307.561
- HiGHS vremena: < 1 s / 7,7 s / 185 s
- GA kvalitet (prosek): 98,7% (najbolje pokretanje 99,0%) / 98,8% / 97,1%
- Slučajna pretraga (prosek): 82,9% / 49,6% / 88,0%
- Poređenje sa slučajnom pretragom pri istom budžetu (slajd 7): 98,3 / 98,5 / 97,1 naspram 82,9 / 49,6 / 88,0
- Početna populacija: 81% / 49% / 88%
- GA na velikoj instanci: 3,4 h po pokretanju, 6 pokretanja
- T = 2880 min, $p_c = 0{,}8$, $p_m = 0{,}15$

**Tri rečenice koje ne smeš pogrešiti:**
1. Slučajna pretraga koristi **istu** popravku, zato razlika meri doprinos evolutivnih operatora.
2. HiGHS je na sve tri ispitane instance bio i bolji i brži — GA je dopuna, ne zamena.
3. Doprinos popravke zavisi od instance (49–88%) i ne može se uopštiti.

**Ako ne znaš odgovor:** reci šta jesi izmerio, gde je granica tog merenja i kako bi to proverio. To je bolji odgovor od nagađanja.
