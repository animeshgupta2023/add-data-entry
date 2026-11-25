# Newspaper advertisement image dataset

50 real, cropped Indian newspaper advertisement images for the
AI-Powered Advertisement Data Entry System POC.

## Contents
- `images/` - 50 JPG files (`ad_001.jpg` ... `ad_050.jpg`)
- `metadata.csv` - filename, source, language, brand/publication, original archive path

## Composition
- 30 English brand ads from Advert Gallery (automotive, banking, jewellery,
  FMCG, electronics, travel, fashion) - useful for brand/headline/price/offer extraction.
- 20 regional-language e-paper ads: Hindi, Marathi, Bengali, Tamil, Telugu,
  Kannada, Malayalam, Gujarati, Odia, Urdu, Assamese (2 each except Urdu: 1).

## Source and attribution
MAdVerse: A Hierarchical Dataset of Multi-Lingual Ads from Diverse Sources and
Categories (IIIT Hyderabad, WACV 2024).
- Dataset: https://zenodo.org/records/10657763  (DOI 10.5281/zenodo.10657763)
- Project page: https://madverse24.github.io/
- Code: https://github.com/Amruth-sagar/MAdVerse

Images were extracted from the MAdVerse archives with HTTP range requests so
only these 50 files were downloaded.

## Licence / usage
Dataset annotations are CC BY-NC 4.0. The MAdVerse consortium does not hold
copyright over the underlying images; use is intended for non-commercial
research and testing. Please respect the original publishers' terms.
