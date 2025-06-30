import json
import sys
from collections import OrderedDict

from deep_translator import GoogleTranslator
from SPARQLWrapper import JSON, SPARQLWrapper

endpoint_url = "https://query.wikidata.org/sparql"

query = """
SELECT ?country ?countryLabel ?language ?languageLabel ?languageCode ?citizenshipLabel ?countryCode WHERE {
  VALUES ?country {
    # Council of Europe member states
    wd:Q222 wd:Q228 wd:Q227 wd:Q399 wd:Q31 wd:Q33 wd:Q40 wd:Q218 wd:Q219 wd:Q224 wd:Q225 wd:Q233
    wd:Q236 wd:Q213 wd:Q214 wd:Q217 wd:Q229 wd:Q238 wd:Q41 wd:Q28 wd:Q189 wd:Q15304003 wd:Q43 wd:Q212
    wd:Q183 wd:Q191 wd:Q142 wd:Q145 wd:Q159 wd:Q403 wd:Q38 wd:Q230 wd:Q221 wd:Q27 wd:Q215 wd:Q347
    wd:Q211 wd:Q217 wd:Q233 wd:Q31 wd:Q40 wd:Q214 wd:Q38 wd:Q35 wd:Q20 wd:Q29999 wd:Q37 wd:Q32
    wd:Q29 wd:Q218 wd:Q36 wd:Q45 wd:Q184 wd:Q1246 wd:Q235 wd:Q34 wd:Q39 wd:Q212
  }

  OPTIONAL {
    ?country wdt:P37 ?language.  # Official language(s)
    ?language wdt:P424 ?languageCode.  # Language code
  }
  OPTIONAL { ?country wdt:P297 ?countryCode. }  # ISO 3166-1 alpha-2 code (e.g., IT, SI)
  OPTIONAL {
    ?country wdt:P1549 ?citizenshipLabel.  # Demonym (citizenship)
    FILTER(LANG(?citizenshipLabel) = "en").
  }

  SERVICE wikibase:label {
    bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en".
  }
}
ORDER BY ASC(?countryLabel)
"""


def get_results(endpoint_url, query):
    user_agent = "WDQS-example Python/%s.%s" % (
        sys.version_info[0],
        sys.version_info[1],
    )
    # TODO adjust user agent; see https://w.wiki/CX6
    sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()


results = get_results(endpoint_url, query)
langs_code = set()
countries_langs = OrderedDict()

cnt = 0
cnt_countries = 0
for result in results["results"]["bindings"]:
    try:
       
        country = result["countryLabel"]["value"]

        country = "Czechia" if country == "Czech Republic" else country
        country = "Netherlands" if country == "Kingdom of the Netherlands" else country
        country = "Bosnia & Herzegovina" if country == "Bosnia and Herzegovina" else country
        #citizenship = "Turkish" if country == "Turkey" else result["citizenshipLabel"]["value"]
        #citizenship = "Ukrainian" if country == "Ukraine" else result["citizenshipLabel"]["value"]
        lang_code = result["languageCode"]["value"]
        language = result["languageLabel"]["value"]
        
        #translator = GoogleTranslator(source="en", target=lang_code)
        langs_code.add(lang_code)

        # if language == "Hebrew":
        #     print(country, citizenship, lang_code, language)

        if country not in countries_langs:
            countries_langs[country] = {
                "COUNTRY_ID": result["countryCode"]["value"],
                "citizenships": set(),
                "languages": set(),
                "languages_code": set(),
            }
            cnt_countries += 1

        # if citizenship[-1] == "s":
        #     citizenship = citizenship[:-1]

        # if citizenship[-4:] != "land":
        #     countries_langs[country]["citizenships"].add(citizenship)

        countries_langs[country]["languages"].add(language)

        countries_langs[country]["languages_code"].add(lang_code)

        cnt += 1

    except Exception as e:
        print("#" * 10)
        print("language code not found", language, lang_code)
        continue

print("Total languages found: ", cnt)
print("Total unique languages found: ", len(langs_code))

for country in countries_langs:
    countries_langs[country]["citizenships"] = list(
        countries_langs[country]["citizenships"]
    )
    countries_langs[country]["languages"] = list(countries_langs[country]["languages"])
    countries_langs[country]["languages_code"] = list(
        countries_langs[country]["languages_code"]
    )

    if len(countries_langs[country]["citizenships"]) > 1:
        print(country, countries_langs[country]["citizenships"])


print(countries_langs)
print("Total countries: ", len(countries_langs))

with open("data/countries_langs.json", "w") as f:
    json.dump(countries_langs, f, indent=4)
