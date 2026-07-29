# Specifikacija prezentacije za odbranu master rada

## Komunikacioni cilj

Do kraja izlaganja komisija treba da razume problem optimalne raspodele poslova na heterogene računare, način na koji je razvijen genetički algoritam i zašto rezultati pokazuju da je on kvalitetna i fleksibilna dopuna, ali ne i zamena za HiGHS na ispitanim linearnim instancama.

## Publika, format i trajanje

- Publika: komisija na formalnoj odbrani master rada na Elektrotehničkom fakultetu Univerziteta u Beogradu.
- Jezik: srpski, latinica.
- Format: PowerPoint prezentacija, 16:9.
- Obim: 11 slajdova.
- Trajanje: približno 10 minuta.
- Prateći materijal: zaseban tekst govora, organizovan po slajdovima i usklađen sa prikazanim sadržajem.

## Narativ

Prezentacija prati tok: problem i motivacija → matematički model → doprinos rada → predloženi algoritam → eksperimentalna evaluacija → tumačenje rezultata → zaključak. Akcenat je na eksperimentalnim rezultatima i njihovom značenju, dok se matematički i algoritamski detalji prikazuju samo u meri potrebnoj za razumevanje doprinosa.

## Struktura slajdova

1. **Optimalna raspodela poslova na računarima**  
   Minimalan naslovni slajd sa ETF identitetom, imenom kandidata, mentorima i datumom.

2. **Heterogeni računari pretvaraju raspodelu poslova u optimizacioni problem**  
   Servisi, računari, različita vremena izvršavanja i zarade, ograničeni kapacitet i tržišni zahtev. Kratak primer primene u cloud računarstvu ili industrijskoj proizvodnji.

3. **Cilj je maksimalna zarada uz dva skupa ograničenja**  
   Sažeta ILP formulacija: funkcija cilja, ograničenje vremena računara, ograničenje zahteva servisa i celobrojnost odluka. Bez izvođenja i bez dodatnih teorijskih detalja.

4. **Rad povezuje egzaktni model i fleksibilnu metaheuristiku**  
   Ciljevi i doprinosi: formulacija problema, genetički algoritam, specijalizovana procedura popravke, Python i C++ implementacije i evaluacija na tri veličine.

5. **Genetički algoritam usmerava pretragu kroz evolutivnu petlju**  
   Jednostavan tok: inicijalizacija → selekcija i elitizam → ukrštanje i mutacija → popravka → nova generacija. Istaknuti turnirsku selekciju, elitizam i injekciju raznolikosti.

6. **Procedura popravke garantuje dopustivost i lokalno poboljšava rešenje**  
   Četiri faze: uklanjanje nedozvoljenih vrednosti, poštovanje zahteva, oslobađanje prekoračenog vremena i pohlepno popunjavanje najprofitabilnijim jedinicama. Ovo je centralni algoritamski doprinos.

7. **Eksperimenti proveravaju kvalitet i skaliranje na tri nivoa**  
   Instance 10×10, 100×100 i 1000×1000; HiGHS kao referenca; slučajna pretraga kao kontrola; više nezavisnih pokretanja; isti hardver.

8. **GA dostiže 97–99% optimuma, ali HiGHS ostaje najefikasniji**  
   Jasno poređenje rezultata za tri instance. Naglasiti: GA 98,7%, 98,8% i 97,1%; HiGHS 100% i kraće vreme na svim ispitanim linearnim instancama.

9. **Evolutivni operatori donose veliki dobitak u odnosu na slučajnu pretragu**  
   Poređenje GA i slučajne pretrage: 98,7% prema 83,8%; 98,1% prema 49,6%; 97,0% prema 88,0%. Koristiti grafikon ili jednostavno vizuelno poređenje zasnovano na podacima iz rada.

10. **Rezultati određuju gde GA ima smisla, a gde nema**  
    HiGHS je preporučen za ispitane linearne instance. GA je opravdan kada su model, ograničenja ili funkcija cilja nelinearni, dinamički ili teško izrazivi ILP formulacijom. Navesti ograničenja evaluacije: jedna instanca po veličini i višesatno izvršavanje velike GA konfiguracije.

11. **Genetički algoritam je kvalitetna dopuna egzaktnom rešavanju**  
    Završna sinteza: specijalizovana popravka + evolucija daju stabilna rešenja visokog kvaliteta; GA ne zamenjuje HiGHS; pravci daljeg rada su adaptivni operatori, lokalna pretraga, dinamičke i višekriterijumske varijante.

## Vizuelni sistem

- Formalan, akademski izgled sa tamnoplavom, belom i diskretnom svetloplavom paletom usklađenom sa ETF identitetom.
- Minimalan naslovni slajd; sadržajni slajdovi imaju velike zaključne naslove i malo teksta.
- Koristiti postojeći ETF logo i grafikone iz rada tamo gde direktno podržavaju poruku.
- Koristiti samo jednu jednostavnu šemu algoritma; bez dekorativnih dijagrama i bez pretrpanih tabela.
- Naslovi najmanje 35 pt, naslov prezentacije najmanje 50 pt, tekst najmanje 16 pt.
- Govorni tekst i vremenske napomene ne prikazuju se na samim slajdovima.

## Raspodela vremena

- Slajd 1: 20 sekundi
- Slajdovi 2–4: ukupno oko 2 minuta i 30 sekundi
- Slajdovi 5–6: ukupno oko 2 minuta
- Slajdovi 7–9: ukupno oko 3 minuta
- Slajdovi 10–11: ukupno oko 2 minuta

Ukupno ciljano trajanje je približno 9 minuta i 50 sekundi, uz malu rezervu za prelaze.

## Izlazni fajlovi

- `prezentacija-odbrana-master-rada.pptx`
- `tekst-za-izlaganje.md`

PowerPoint mora biti renderovan i vizuelno proveren slajd po slajd pre isporuke. Govorni tekst treba da zvuči prirodno, bez čitanja sadržaja sa slajda reč po reč.
