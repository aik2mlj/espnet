All the scripts are located under `spk1/local/`.

- Silence detection on vocal stems: 730k ⇒ 620k
    - `12m_split_on_silence.py`
- Data cleaning
    - match local filenames with gcs_link in csv
    - filter: currently filtering out another ~9% of data (61742 songs)
        - everything containing “feat.”, “&”, etc. that indicate multiple singers
        - remove all classical related (Keyword: Orchestra, philharmonic, symphony, etc.)
        - remove all DJ
        - remove all multiple artist in artist name
        - remove unknown / anonymous
    - Some concerns:
        - Weird artist names with only 1 song? (there are quite a lot of artists with non-English names,
            - e.g.: –ó–ª–∞—Ç–∞ –û–≥–Ω–µ–≤–∏—á
        - Found some Chinese singers with Both their Chinese name and English name in dataset
            - What to do? List some common ones and manually merge? Throw all Chinese (and other non-English) away?
        - some DJ / electronic artist names (like rezz) not filtered out with current method, still noisy

Filter result:
- Number of unique artists left (case insensitive): 59190/93422 artists before filter & sad
    - corresponds to ~ 90% of songs after singing activity detection
