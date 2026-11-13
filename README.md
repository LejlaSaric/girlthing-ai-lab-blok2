# GirlTHing AI Lab — Blok 2: AI / LLM

Materijali za radionice 5–8. Kroz sve četiri radionice gradimo isti primjer: **"Studentski asistent"**, RAG chatbot koji odgovara na pitanja o CV-u, razgovoru za posao, praksama, stipendijama, učenju i o samom programu. Nakon radionica timovi ovaj kod prilagođavaju svojoj temi.

## Sadržaj

| Datoteka / folder | Šta je | Radionica |
| --- | --- | --- |
| `notebooks/R5_LLM_i_prompting.ipynb` | Prvi pozivi API-ja, tokeni, prompt engineering, mini chatbot | R5 |
| `notebooks/R6_RAG_i_baza_znanja.ipynb` | Embeddings, chunking, pretraga, RAG od nule | R6 |
| `notebooks/R7_kvalitet_i_sigurnost.ipynb` | Streaming, evaluacija, prompt injection, troškovi | R7 |
| `notebooks/*_rjesenja.ipynb` | Iste vježbe s rješenjima (za edukatoricu) | R5–R7 |
| `data/` | Primjer baze znanja (6 dokumenata) | R6–R8 |
| `rag.py` | RAG logika: učitavanje, chunking, embeddings, pretraga, odgovor | R6–R8 |
| `app.py` | Streamlit web aplikacija | R7–R8 |
| `eval.py`, `test_set.json` | Automatska evaluacija na 10 test pitanja | R7 |
| `DEPLOY.md` | Korak po korak objava na Streamlit Community Cloud + šablon prezentacije | R8 |

## Pokretanje

### Notebookovi (Google Colab)

1. Otvori notebook u Colabu (File → Open notebook → GitHub, ili upload datoteke).
2. Lijevo klikni ikonu ključa (Secrets) i dodaj `OPENAI_API_KEY` sa svojim ključem. Uključi "Notebook access".
3. Pokreni ćelije redom.

### Aplikacija lokalno (VS Code ili GitHub Codespaces)

```bash
pip install -r requirements.txt
cp .env.example .env          # pa upiši svoj ključ u .env
python rag.py "Koliko dug treba biti CV?"   # brzi test u terminalu
streamlit run app.py          # web aplikacija na http://localhost:8501
python eval.py                # evaluacija na test setu
```

Ko nema dovoljno jak laptop može sve raditi u **GitHub Codespaces** (besplatna mjesečna kvota uz GitHub račun): na stranici repozitorija Code → Codespaces → Create codespace.

## Prilagodba za timski projekat

1. Obriši primjere iz `data/` i stavi svoje dokumente (.md, .txt ili .pdf).
2. U `rag.py` promijeni `SYSTEM_PROMPT`: ime chatbota, temu i ton.
3. U `app.py` promijeni naslov i opis u bočnoj traci.
4. Napiši vlastiti `test_set.json` (najmanje 10 pitanja, uključujući 1–2 na koja chatbot NE smije znati odgovor).
5. Pokreni `python eval.py` i poboljšavaj prompt i dokumente dok rezultat ne bude dobar.

## Modeli i troškovi

- Chat model: `gpt-6-luna` ($0.10 / 1M ulaznih tokena, $0.50 / 1M izlaznih).
- Embeddings: `text-embedding-3-small`.
- Jedno pitanje s RAG kontekstom ima oko 1.500 ulaznih tokena, pa 1.000 pitanja košta oko $0.25.

Cijene i nazivi modela provjereni su u novembru 2026. na developers.openai.com. Ako se promijene, mijenja se samo `CHAT_MODEL` u `rag.py`.

## Sigurnost

- API ključ **nikad** ne ide u kod, notebook ili na GitHub. Koristi `.env`, Colab Secrets ili Streamlit Secrets.
- Svaki tim ima svoj OpenAI projekat s vlastitim ključem i mjesečnim limitom potrošnje.
