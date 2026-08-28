# Tekst za izlaganje — odbrana master rada

Procenjeno ukupno trajanje: približno 10–11 minuta.

## Slajd 1 — Naslov (0:25)

Dobar dan. Ja sam Mihailo Radojević i predstaviću master rad pod nazivom „Optimalna raspodela poslova na računarima“. Rad je urađen pod mentorstvom profesora Dragana Olćana i docentkinje Jovane Petrović. Osnovno pitanje rada je kako rasporediti veliki broj poslova na heterogene računare tako da ukupna zarada bude što veća, uz poštovanje svih vremenskih i tržišnih ograničenja.

## Slajd 2 — Problem i matematički model staju u jedan matrični zapis (1:05)

Posmatramo više servisa i heterogene računare. Oznaka p-i-j predstavlja zaradu po jedinici servisa i na računaru j, t-i-j vreme izvršavanja te jedinice na računaru j, a d-i ukupan zahtev za servis i. Promenljiva x-i-j označava broj jedinica servisa i raspoređenih na računar j. Maksimizujemo ukupnu zaradu uz dva glavna ograničenja: nijedan računar ne sme prekoračiti raspoloživo vreme, a za svaki servis ne sme se realizovati više jedinica nego što je zahtevano. Promenljive su nenegativne i celobrojne, pa formulacija ima m puta n promenljivih.

## Slajd 3 — Rad povezuje matematički model i evolutivnu pretragu (1:00)

Rad povezuje matematički model i evolutivnu pretragu. Specijalizacija genetičkog algoritma odnosi se na predstavljanje rešenja i postupak popravke prilagođen ovom problemu. Hromozom je matrica raspodele poslova, a posle inicijalizacije, ukrštanja i mutacije primenjuje se namenska četvorofazna popravka. Selekcija, elitizam, ukrštanje i mutacija jesu standardni elementi GA, dok su matrični zapis hromozoma i popravka vezani za razmatranu raspodelu poslova.

## Slajd 4 — Popravka održava dopustivost i poboljšava rešenje (1:05)

Procedura popravke je centralni element algoritma i ima četiri determinističke faze. Najpre uklanja negativne vrednosti. Zatim smanjuje količine kada je premašen zahtev servisa. U trećoj fazi oslobađa prekoračeno vreme računara uklanjanjem najmanje profitabilnih jedinica, a na kraju preostali kapacitet pohlepno popunjava najprofitabilnijim dopustivim poslovima. GA i dalje ostaje populacioni algoritam globalne pretrage; popravka samo održava dopustivost i lokalno doteruje pojedinačna rešenja koja evolutivna pretraga proizvodi.

## Slajd 5 — Ista formulacija je ispitana na tri problema različite veličine (0:50)

Ista matematička formulacija ispitana je na tri problema: 10 puta 10, 100 puta 100 i 1000 puta 1000, odnosno sa sto, deset hiljada i milion promenljivih. HiGHS je korišćen kao referentni alat za sve tri lokalno rešene instance, a slučajna pretraga kao kontrolna metoda. Za dodatnu online proveru korišćen je NEOS server sa Gurobi solverom. Online solver rešio je probleme 10 puta 10 i 100 puta 100, ali problem sa milion promenljivih nije mogao da reši tim putem. Konfiguracije su pokretane više puta, a sva vremenska merenja izvršena su na istom računaru, MacBook Pro uređaju sa M1 Pro procesorom i 16 gigabajta memorije.

## Slajd 6 — GA dostiže 97–99% optimuma, ali HiGHS ostaje najefikasniji (1:15)

Na grafikonu su prikazane reprezentativne konfiguracije genetičkog algoritma. Na maloj instanci konfiguracija sa istim budžetom kao slučajna pretraga dostiže 98,7 procenata optimuma; uz veći budžet GA prelazi 99 procenata, ali taj rezultat nije korišćen u ovom direktnom poređenju. Na srednjoj instanci GA dostiže 98,8 procenata, a na velikoj 97,1 procenat. To su kvalitetna i stabilna rešenja, ali je ključan nalaz da je HiGHS na sve tri ispitane linearne instance bio i bolji i brži. Malu instancu rešava za manje od sekunde, srednju za oko 7,7 sekundi, a na velikoj instanci sa milion promenljivih dokazuje optimum za oko 185 sekundi. GA na velikoj instanci zahteva više sati. Dakle, na ovom modelu genetički algoritam nije zamena za egzaktni alat.

## Slajd 7 — Evolutivni operatori donose veliki dobitak u odnosu na slučajnu pretragu (0:55)

Ovde su prikazane GA konfiguracije odabrane za poređenje sa kontrolnom metodom, pa se pojedine vrednosti blago razlikuju od prethodnog slajda. GA je bolji od slučajne pretrage na sve tri ispitane instance, ali veličina razlike zavisi od instance. Posebno odstupa instanca 100 puta 100. Velika razlika na njoj nije posledica broja evaluacija, jer su obe metode imale isti budžet od deset miliona evaluacija. Obe koriste isto generisanje početnih rešenja i istu popravku, pa razlika pokazuje doprinos selekcije, ukrštanja, mutacije i elitizma, koji pretragu usmeravaju ka kvalitetnijim raspodelama.

## Slajd 8 — Uticaj veličine populacije na konvergenciju pri istom broju evaluacija (0:50)

Pri istom ukupnom broju od milion evaluacija populacija od 1000 jedinki brže napreduje na početku jer prolazi kroz više generacija. Populacija od 2000 jedinki napreduje sporije, ali zahvaljujući većoj raznovrsnosti na kraju dostiže nešto bolju prosečnu vrednost: 88.817 prema 88.657 dinara. Dakle, veličina populacije utiče na konvergenciju: menja odnos između broja jedinki u generaciji, broja generacija i raznovrsnosti pretrage.

## Slajd 9 — Veća populacija sve manje povećava vrednost funkcije cilja (0:55)

Na problemu 100 puta 100 povećanje populacije povećava prosečnu vrednost funkcije cilja, ali se veličina dodatnog poboljšanja smanjuje. Prelazak sa 100 hiljada na 200 hiljada jedinki povećava prosečnu vrednost funkcije cilja za približno 0,15%, dok se prosečno vreme izvršavanja povećava oko tri i po puta. Zato najveća populacija nije automatski najbolji inženjerski izbor; treba birati tačku posle koje dodatni kvalitet više ne opravdava dodatno vreme.

## Slajd 10 — Ispitane linearne instance favorizuju HiGHS (0:55)

Eksperimentalni zaključak rada odnosi se na ispitane statičke linearne instance: HiGHS je dao najbolju vrednost funkcije cilja uz najmanje vreme. Nelinearnosti, dinamičke promene, više ciljeva i ograničenja koja se teško izražavaju linearnom formulacijom nisu ispitivani u radu; navedeni su kao moguća proširenja matematičkog modela i pravci buduće primene GA. Na grafikonu se vidi da je šest pokretanja velike instance dalo veoma slične putanje, ali jedno takvo pokretanje traje oko 3,4 sata. Ograničenje evaluacije je i to što je korišćen po jedan problem svake veličine.

## Slajd 11 — Tri nalaza opisuju performanse genetičkog algoritma (0:35)

Rezultati daju tri odgovora. Prvo, GA pronalazi stabilna rešenja kvaliteta od približno 97 do 99 procenata referentnog optimuma. Drugo, GA je bolji od slučajne pretrage na sve tri ispitane instance, ali veličina razlike zavisi od instance i posebno je izražena na problemu 100 puta 100. Treće, povećanje populacije povećava vrednost funkcije cilja, ali uz sve manji dodatni dobitak i brzo rastuće vreme izvršavanja.

## Slajd 12 — Zaključak (0:40)

Jedan od glavnih doprinosa rada je specijalizovana procedura popravke: posle svake izmene hromozoma vraća rešenje u dopustivo stanje i zatim ga lokalno poboljšava profitabilnijim dodelama. HiGHS ostaje prvi izbor za ispitane linearne instance. Primena GA na proširene, dinamičke i višekriterijumske formulacije nije ispitana u ovom radu i predstavlja pravac daljeg istraživanja, zajedno sa adaptivnim operatorima i lokalnom pretragom. Hvala na pažnji.
