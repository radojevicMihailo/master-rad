# Tekst za izlaganje — odbrana master rada

Procenjeno trajanje: oko 7,5 minuta govora.

## Slajd 1 — Naslov (0:25)

Dobar dan. Ja sam Mihailo Radojević i predstaviću master rad „Optimalna raspodela poslova na računarima“, urađen pod mentorstvom profesora Dragana Olćana i docentkinje Jovane Petrović. Pitanje koje rad postavlja je jednostavno: kako veliki broj poslova rasporediti na skup računara različitih performansi tako da ukupna zarada bude što veća, a da pritom nijedno vremensko ni tržišno ograničenje ne bude prekršeno.

## Slajd 2 — Problem i matematički model staju u jedan matrični zapis (0:40)

Na raspolaganju imamo više servisa i više računara, koji se razlikuju po brzini izvršavanja. Veličina p-i-j je zarada po jedinici servisa i na računaru j, t-i-j je vreme izvršavanja te jedinice, a d-i je broj jedinica servisa i koje treba realizovati. Promenljiva x-i-j kaže koliko je jedinica servisa i dodeljeno računaru j. Maksimizujemo ukupnu zaradu uz dva ograničenja: nijedan računar ne sme da prekorači svoje raspoloživo vreme, i ni za jedan servis ne sme se realizovati više jedinica nego što je traženo. Promenljive su celobrojne i nenegativne, pa problem ima m puta n promenljivih.

## Slajd 3 — Rad povezuje matematički model i evolutivnu pretragu (0:30)

Rad povezuje ovaj matematički model sa evolutivnom pretragom. Ono što je ovde specifično nije sam genetički algoritam, nego način na koji je rešenje predstavljeno i postupak popravke prilagođen ovom problemu. Hromozom je matrica raspodele poslova, a posle inicijalizacije, ukrštanja i mutacije primenjuje se namenska četvorofazna popravka. Selekcija, elitizam, ukrštanje i mutacija su standardni elementi; matrični zapis hromozoma i popravka su vezani za razmatranu raspodelu poslova.

## Slajd 4 — Popravka održava dopustivost i poboljšava rešenje (0:30)

Popravka je centralni element algoritma i ima četiri determinističke faze. Prva uklanja negativne vrednosti. Druga smanjuje količine tamo gde je premašen zahtev servisa. Treća oslobađa prekoračeno vreme računara tako što uklanja najmanje profitabilne jedinice. Četvrta preostali kapacitet pohlepno popunjava najprofitabilnijim dopustivim poslovima. Genetički algoritam pritom ostaje populaciona globalna pretraga — popravka samo vraća rešenje u dopustivo stanje i lokalno ga dotera.

## Slajd 5 — Ista formulacija je ispitana na tri problema različite veličine (0:40)

Ista formulacija ispitana je na tri problema: deset puta deset, sto puta sto i hiljadu puta hiljadu, odnosno sa sto, deset hiljada i milion promenljivih. HiGHS je korišćen kao referentni alat, a slučajna pretraga kao kontrolna metoda. Za dodatnu proveru korišćen je i NEOS server sa Gurobi solverom; on je rešio prve dve instance, ali problem sa milion promenljivih tim putem nije mogao da se reši. Svaka konfiguracija je pokretana više puta, a sva merenja vremena urađena su na istom računaru — MacBook Pro sa M1 Pro procesorom i šesnaest gigabajta memorije.

## Slajd 6 — GA dostiže 97–99% optimuma, ali HiGHS ostaje najefikasniji (0:55)

Grafikon prikazuje najbolje konfiguracije genetičkog algoritma na svakoj instanci; sve navedene vrednosti su proseci pokretanja, ne najbolja pojedinačna pokretanja. Na maloj instanci, sa milion evaluacija, algoritam dostiže 98,7 procenata optimuma; još veća konfiguracija diže prosek na 98,9, a najbolje pokretanje prelazi 99 procenata. Na srednjoj instanci prosek je 98,8, a na velikoj 97,1 procenat. To su kvalitetna i stabilna rešenja. Ključni nalaz je ipak drugi: na sve tri ispitane linearne instance HiGHS je bio i bolji i brži. Malu instancu rešava za manje od sekunde, srednju za oko 7,7 sekundi, a na velikoj instanci sa milion promenljivih dokazuje optimum za oko 185 sekundi. Genetičkom algoritmu je na toj instanci potrebno preko četiri sata. Dakle, na ovom modelu on nije zamena za egzaktni alat.

## Slajd 7 — Evolutivni operatori donose veliki dobitak u odnosu na slučajnu pretragu (0:50)

Ovde su obe metode poređene pri istom broju evaluacija na svakoj instanci — sto hiljada na maloj, deset miliona na srednjoj i dva miliona na velikoj — i prikazani su proseci pokretanja. Genetički algoritam je bolji od slučajne pretrage na sve tri instance, ali koliko je bolji zavisi od instance: 98,3 prema 82,9 procenata na maloj, 98,5 prema 49,6 na srednjoj i 97,1 prema 88,0 na velikoj. Najveća razlika je na problemu sto puta sto. Ta razlika ne dolazi od broja evaluacija, jer je budžet isti, a obe metode koriste i isto generisanje početnih rešenja i istu popravku. Ono što ostaje kao objašnjenje jeste doprinos selekcije, ukrštanja, mutacije i elitizma, koji pretragu usmeravaju ka kvalitetnijim raspodelama.

## Slajd 8 — Uticaj veličine populacije na konvergenciju pri istom broju evaluacija (0:25)

Uz isti ukupan broj od milion evaluacija, populacija od hiljadu jedinki brže napreduje na početku, jer prolazi kroz više generacija. Populacija od dve hiljade napreduje sporije, ali zahvaljujući većoj raznovrsnosti na kraju stiže do nešto bolje prosečne vrednosti — 88.817 prema 88.657 dinara. Veličina populacije, dakle, menja odnos između broja jedinki, broja generacija i raznovrsnosti pretrage.

## Slajd 9 — Vreme izvršavanja raste mnogo brže od kvaliteta rešenja (0:55)

Na instanci sto puta sto konfiguracije su poređane po ukupnom broju evaluacija, od pola miliona do četrdeset miliona; ispod svake oznake stoje veličina populacije i broj generacija čiji je proizvod taj budžet. Levo je prosečan kvalitet, desno prosečno vreme jednog pokretanja. Vreme raste gotovo linearno sa brojem evaluacija, a kvalitet se brzo zasićuje: već pri deset miliona evaluacija imamo 98,5 procenata optimuma. Poslednji korak to pokazuje najjasnije — četiri puta više posla podiže kvalitet za svega 0,15 procenata, a vreme sa petsto na hiljadu sedamsto osamdeset sekundi. Dve konfiguracije sa istih deset miliona evaluacija izdvajaju efekat populacije: sto hiljada jedinki daje bolji rezultat od pedeset hiljada, ali je skuplje po evaluaciji. Zato najveća konfiguracija nije automatski i najbolji inženjerski izbor.

## Slajd 10 — Ispitane linearne instance favorizuju HiGHS (0:35)

Eksperimentalni zaključak važi za ispitane statičke linearne instance: HiGHS je dao najbolju vrednost funkcije cilja, i to za najkraće vreme. Nelinearnosti, dinamičke promene, više ciljeva i ograničenja koja se teško zapisuju linearno nisu ispitivani; navedeni su kao moguća proširenja modela i pravci primene genetičkog algoritma. Na grafikonu se vidi da je šest pokretanja velike instance dalo veoma slične putanje — ali jedno takvo pokretanje traje oko 3,4 sata. Ograničenje evaluacije je i to što je korišćen po jedan problem svake veličine.

## Slajd 11 — Tri nalaza opisuju performanse genetičkog algoritma (0:25)

Rezultati daju tri odgovora. Prvo, genetički algoritam nalazi stabilna rešenja kvaliteta od približno 97 do 99 procenata referentnog optimuma. Drugo, bolji je od slučajne pretrage na sve tri instance, a razlika je najizraženija na problemu sto puta sto. Treće, veći računski budžet daje bolju vrednost funkcije cilja, ali uz sve manji dobitak: četiri puta više evaluacija donosi 0,15 procenata bolje rešenje, a traje tri i po puta duže.

## Slajd 12 — Zaključak (0:30)

Glavni doprinos rada je specijalizovana procedura popravke: posle svake izmene hromozoma ona vraća rešenje u dopustivo stanje i zatim ga poboljšava profitabilnijim dodelama. Za ispitane linearne instance HiGHS ostaje prvi izbor. Primena genetičkog algoritma na proširene, dinamičke i višekriterijumske formulacije nije ispitana u ovom radu i predstavlja pravac daljeg istraživanja, zajedno sa adaptivnim operatorima i lokalnom pretragom. Hvala na pažnji.
