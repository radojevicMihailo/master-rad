# Dizajn drugog dela prezentacije: performanse genetičkog algoritma

## Cilj

Drugi deo prezentacije treba da predstavi glavne rezultate kao povezanu priču: koliko je GA kvalitetan, zašto nadmašuje slučajnu pretragu, kako veličina populacije utiče na konvergenciju i kolika je cena dodatnog kvaliteta. Komisija na kraju treba da razume ne samo krajnje procente već i praktični kompromis između kvaliteta, vremena i izbora hiperparametara.

## Obim

- Prezentacija se proširuje sa 12 na 14 slajdova.
- Prvih sedam slajdova ostaje sadržajno nepromenjeno.
- Drugi deo se reorganizuje, uz dva nova slajda o performansama GA.
- Numeracija se ažurira na `1/14`–`14/14`.
- Ciljano trajanje ostaje približno 10–11 minuta; novi slajdovi se izlažu sažeto, po oko 40–50 sekundi.

## Narativ slajdova 7–14

### 7. Eksperimentalna postavka

Uvesti tri veličine problema, referentni rezultat HiGHS-a, slučajnu pretragu, više pokretanja i isti hardver.

### 8. Kvalitet GA u odnosu na HiGHS

Prikazati da GA dostiže približno 97–99% referentnog rezultata, ali da je HiGHS brži i bolji na ispitanim linearnim instancama. Y-osa mora biti jasno označena kao procenat referentnog optimuma.

### 9. Doprinos evolutivnih operatora

Porediti GA i slučajnu pretragu. Za instancu 100×100 koristiti metodološki čisto poređenje sa istih 10 miliona evaluacija:

- GA, `P=50.000`, `G=200`: 1.994.685, odnosno 98,59% optimuma;
- slučajna pretraga: 1.003.308, odnosno 49,59% optimuma.

Na slajdu eksplicitno navesti da obe metode koriste istu proceduru popravke i isti broj evaluacija za prikazano poređenje 100×100.

### 10. Uticaj veličine populacije pri istom broju evaluacija

Koristiti grafikon `rad/slike/compare-10x10-pop.pdf`, koji poredi `P=1000, G=1000` i `P=2000, G=500` pri milion evaluacija.

Poruke:

- manja populacija brže napreduje jer prolazi kroz više generacija;
- veća populacija održava veću raznovrsnost i dostiže nešto bolji konačni rezultat;
- pri fiksnom budžetu postoji kompromis između broja jedinki i broja evolutivnih ciklusa.

### 11. Opadajući prinos povećanja populacije

Prikazati rezultate za 100×100 kombinovanim grafikonom:

- stubići: kvalitet rešenja kao procenat optimuma;
- linija: prosečno vreme jednog pokretanja;
- reprezentativne populacije: 5.000, 10.000, 50.000, 100.000, 150.000 i 200.000.

Poruke:

- rast populacije poboljšava kvalitet i stabilnost;
- dobitak se postepeno smanjuje;
- prelazak sa 100.000 na 200.000 jedinki daje približno 0,15 procentnih poena u proseku, uz oko 3,5 puta duže izvršavanje.

### 12. Gde GA ima smisla

Iz prethodnih rezultata izvesti preporuku:

- HiGHS prvo za statičke linearne instance ovog tipa;
- GA za nelinearne, dinamičke, višekriterijumske modele ili ograničenja koja se teško modeluju linearnom formulacijom;
- ograničenje eksperimenta: po jedna instanca svake veličine.

Ispod grafikona zadržati dva kratka zaključka u bullet formi.

### 13. Sinteza performansi

Sažeti tri odgovora:

- kvalitet: 97–99% referentnog optimuma;
- mehanizam: evolutivni operatori značajno nadmašuju slučajnu pretragu;
- cena: povećanje populacije donosi opadajući prinos i mora se opravdati vremenom.

Ovaj slajd služi kao prelaz ka završnom zaključku, bez ponavljanja svih brojki.

### 14. Zaključak

Zadržati tri jasne poruke u ujednačenim karticama:

- GA daje visok kvalitet;
- HiGHS je prvi izbor za ispitane linearne instance;
- vrednost GA je u fleksibilnosti proširenih modela.

## Vizuelni sistem

- Zadržati postojeću tamnoplavu, belu, svetloplavu i zelenu paletu.
- Naslovi 36–44 pt, tekst najmanje 16 pt.
- Svaki rezultatski slajd ima jedan dominantan grafikon i najviše dva kratka zaključka.
- Isti pojmovi koriste iste boje: GA plava, slučajna pretraga siva, HiGHS tamnoplava, praktična preporuka zelena.
- Izbegavati duge pasuse i ponavljanje zaključaka sa prethodnih slajdova.

## Provera sadržaja

- Sve vrednosti se preuzimaju iz sirovih rezultata u direktorijumima `rezultati-*` i proveravaju prema optimumima u radu.
- Na svakom poređenju mora biti naveden broj evaluacija ili jasno naznačeno da budžeti nisu isti.
- Završni PPTX se renderuje i vizuelno proverava slajd po slajd.
- Posebno proveriti čitljivost osa, legendi, oznaka populacije, vremena i numeracije `x/14`.
