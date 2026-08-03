# GA Performance Story Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Izraditi konačnu prezentaciju od 12 slajdova u kojoj je uvod sažet na pet slajdova, a glavni rezultati predstavljeni kao povezana priča o kvalitetu, doprinosu evolutivnih operatora, veličini populacije i ceni izvršavanja.

**Architecture:** Postojeći generator `.tmp/prezentacija-odbrana/build-deck.mjs` ostaje jedini izvor PPTX-a i govornog teksta. Sadržaj se rekonstruiše u 12 slajdova, postojeći vizuelni sistem se zadržava, a svi rezultatski podaci se preuzimaju iz sirovih `rezultati-*` datoteka i postojećih grafikona rada.

**Tech Stack:** JavaScript ES modules, `@oai/artifact-tool`, PowerPoint PPTX, LibreOffice/Poppler za renderovanje, postojeći PDF/PNG grafikoni iz `rad/slike/`.

## Global Constraints

- Prezentacija ima tačno 12 slajdova i numeraciju `1/12`–`12/12`.
- Jezik je srpski, latinica; format 16:9; ciljno trajanje približno 10–11 minuta.
- Prvih pet slajdova pokriva metodologiju, a slajdovi 6–12 rezultate i zaključke.
- Na poređenju 100×100 GA i slučajne pretrage koriste se rezultati sa po 10 miliona evaluacija.
- Svaki rezultatski slajd ima dominantan grafikon i najviše dva kratka zaključka.
- GA je plav, slučajna pretraga siva, HiGHS tamnoplava, praktične preporuke zelene.
- Finalni PPTX se renderuje i vizuelno proverava slajd po slajd.

---

### Task 1: Zaključavanje podataka i izlaznih fajlova

**Files:**
- Modify: `.tmp/prezentacija-odbrana/build-deck.mjs`
- Read: `rezultati-100x100/rezultati/*.txt`
- Read: `rad/main.tex`

**Interfaces:**
- Consumes: sirove vrednosti optimuma, kvaliteta i vremena za populacije 5.000–200.000.
- Produces: konstante podataka u generatoru i izlaze `prezentacija-finalna-ispravljena-v2.pptx` i `tekst-za-izlaganje-ispravljen-v2.md`.

- [ ] **Step 1: Promeniti izlazna imena i zadržati ukupan broj slajdova 12**

U generatoru postaviti:

```js
const OUT_PPTX = path.join(ROOT, "prezentacija-finalna-ispravljena-v2.pptx");
const OUT_SPEECH = path.join(ROOT, "tekst-za-izlaganje-ispravljen-v2.md");
const TOTAL_SLIDES = 12;
```

- [ ] **Step 2: Dodati proverene podatke za 100×100**

U generator dodati jednu strukturu:

```js
const population100 = [
  { population: 5000, quality: 98.0879, seconds: 16.3 },
  { population: 10000, quality: 98.3121, seconds: 30.0 },
  { population: 50000, quality: 98.5912, seconds: 311.4 },
  { population: 100000, quality: 98.7198, seconds: 503.8 },
  { population: 150000, quality: 98.7509, seconds: 1382.6 },
  { population: 200000, quality: 98.8394, seconds: 1778.8 },
];
```

Pre unosa proveriti svako vreme iz odgovarajuće datoteke rezultata i zameniti vrednost ako se razlikuje od prikazane.

- [ ] **Step 3: Proveriti brojke iz izvora**

Run:

```bash
rg -n "PROSECNO VREME|NAJBOLJI|PROSEK" rezultati-100x100/rezultati/ga-cpp-*.txt
```

Expected: svih šest konfiguracija ima čitljive rezultate; vrednosti ugrađene u generator odgovaraju izvornim datotekama.

### Task 2: Sažimanje metodologije na pet slajdova

**Files:**
- Modify: `.tmp/prezentacija-odbrana/build-deck.mjs`

**Interfaces:**
- Consumes: postojeće pomoćne funkcije `addText`, `addPanel`, `addHeader`, `addStepBox`, `addSourceNotes`.
- Produces: slajdove 1–5 i odgovarajućih pet stavki u nizu `speeches`.

- [ ] **Step 1: Zadržati naslovni slajd kao slajd 1**

Sačuvati postojeći naslov, ETF logo, podatke o kandidatu i mentorima; numeracija ostaje `1/12`.

- [ ] **Step 2: Spojiti problem i matematički model u slajd 2**

Levo prikazati motivaciju u najviše tri kratke poruke, a desno funkciju cilja i dva ograničenja. Ukloniti samostalni prethodni slajd matematičkog modela.

- [ ] **Step 3: Spojiti ciljeve rada i evolutivnu petlju u slajd 3**

Levo prikazati tri doprinosa: ILP formulacija, GA sa specijalizovanom popravkom, eksperimentalno poređenje. Desno zadržati tok:

```text
Inicijalizacija → selekcija i elitizam → ukrštanje i mutacija → popravka → nova generacija
```

- [ ] **Step 4: Prenumerisati proceduru popravke i eksperimentalnu postavku**

Postojeći sadržaj procedure popravke postaje slajd 4, a eksperimentalna postavka slajd 5. Ažurirati naslove, imena elemenata, fusere i govor.

- [ ] **Step 5: Proveriti uvodni deo generatorom**

Run:

```bash
node .tmp/prezentacija-odbrana/build-deck.mjs
```

Expected: generator završava bez greške, prijavljuje `Slides: 12`, a `slide-01.png`–`slide-05.png` postoje.

### Task 3: Rekonstrukcija priče o rezultatima

**Files:**
- Modify: `.tmp/prezentacija-odbrana/build-deck.mjs`
- Read: `rad/slike/compare-10x10-pop.pdf`
- Read: `.tmp/prezentacija-odbrana/compare-10x10-pop.png`

**Interfaces:**
- Consumes: `population100`, postojeće rezultate HiGHS/GA/slučajne pretrage i grafikon populacija.
- Produces: slajdove 6–12 i sedam odgovarajućih stavki u `speeches`.

- [ ] **Step 1: Premestiti poređenje GA i HiGHS na slajd 6**

Zadržati stubičasti grafikon i eksplicitnu oznaku `Kvalitet rešenja (% optimuma)`. Ažurirati numeraciju i govor.

- [ ] **Step 2: Ispraviti poređenje GA i slučajne pretrage na slajdu 7**

Za 100×100 koristiti `98,59%` za GA i `49,59%` za slučajnu pretragu, oba pri 10 miliona evaluacija. Dodati kratku oznaku `isti budžet: 10 miliona evaluacija` i zadržati objašnjenje da obe metode koriste istu popravku.

- [ ] **Step 3: Napraviti slajd 8 o veličini populacije**

Koristiti `compare-10x10-pop.png` kao dominantan vizuelni element. Desno prikazati dve poruke: `P=1000: brže rano` i `P=2000: bolje na kraju`, uz napomenu da obe konfiguracije imaju milion evaluacija.

- [ ] **Step 4: Napraviti slajd 9 o opadajućem prinosu**

Napraviti kombinovani grafikon iz `population100`: stubići predstavljaju procenat optimuma, linija prosečno vreme u sekundama. Izdvojiti zaključak da dodatna populacija poboljšava kvalitet uz sve manji marginalni dobitak i znatno veće vreme.

- [ ] **Step 5: Prenumerisati praktičnu preporuku na slajd 10**

Zadržati formulaciju da je HiGHS prvi izbor za ispitane statičke linearne instance, a GA za fleksibilnije modele. Ispod grafikona zadržati tačno dva zaključka.

- [ ] **Step 6: Napraviti sintezu performansi na slajdu 11**

Prikazati tri jednako oblikovane kartice: `97–99% kvalitet`, `evolucija nadmašuje slučajnost`, `veća populacija = opadajući prinos`. Slajd ne ponavlja detaljne grafikone.

- [ ] **Step 7: Zadržati završni zaključak kao slajd 12**

Zadržati tri kartice: visok kvalitet GA, HiGHS prvo, fleksibilnost GA. Ažurirati govor i numeraciju.

### Task 4: Sadržajna, strukturna i vizuelna verifikacija

**Files:**
- Verify: `prezentacija-finalna-ispravljena-v2.pptx`
- Verify: `tekst-za-izlaganje-ispravljen-v2.md`
- Create: `.tmp/prezentacija-odbrana/qa-v2/`

**Interfaces:**
- Consumes: finalni PPTX i govor.
- Produces: potvrdu da prezentacija ima 12 čitljivih i sadržajno ispravnih slajdova.

- [ ] **Step 1: Izgraditi finalne fajlove**

Run:

```bash
node .tmp/prezentacija-odbrana/build-deck.mjs
```

Expected: postoje oba izlazna fajla i generator prijavljuje `Slides: 12`.

- [ ] **Step 2: Proveriti sadržaj i numeraciju**

Run:

```bash
rg -n '"text": "([1-9]|1[0-2])/12"' .tmp/prezentacija-odbrana/slide-*.layout.json
rg -n "10 miliona|98,59|49,59|opadajući prinos" .tmp/prezentacija-odbrana/inspection.ndjson
```

Expected: svih 12 numeracija postoji; ključne metodološke oznake nalaze se u inspekciji.

- [ ] **Step 3: Renderovati PPTX u PDF i JPEG slike**

Run:

```bash
python3 /Users/mihailoradojevic/.codex/plugins/cache/claude-cowork/anthropic-skills/1.0.0/skills/pptx/scripts/office/soffice.py --headless --convert-to pdf --outdir .tmp/prezentacija-odbrana/qa-v2 prezentacija-finalna-ispravljena-v2.pptx
pdftoppm -jpeg -r 150 .tmp/prezentacija-odbrana/qa-v2/prezentacija-finalna-ispravljena-v2.pdf .tmp/prezentacija-odbrana/qa-v2/slide
```

Expected: tačno 12 JPEG slika.

- [ ] **Step 4: Vizuelno pregledati svih 12 slajdova**

Proveriti prelivanje teksta, ose i legende grafikona, čitljivost formula, poravnanje kartica, margine i numeraciju. Svaki uočeni problem ispraviti u generatoru, zatim ponoviti izgradnju i renderovanje.

- [ ] **Step 5: Završna provera arhive**

Run:

```bash
unzip -t prezentacija-finalna-ispravljena-v2.pptx
pdfinfo .tmp/prezentacija-odbrana/qa-v2/prezentacija-finalna-ispravljena-v2.pdf
```

Expected: PPTX arhiva nema greške, a PDF ima `Pages: 12`.
