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
count_languages = []
countries_langs = OrderedDict()

cnt = 0
cnt_countries = 0
for result in results["results"]["bindings"]:
    try:       
        country = result["countryLabel"]["value"]
        
        if country == "country of the Kingdom of the Netherlands":
            continue
        
        if country == "Czech Republic":
            country = "Czechia"
        elif country == "Kingdom of the Netherlands":
            country = "Netherlands"
        elif country == "Bosnia and Herzegovina":
            country = "Bosnia & Herzegovina"
        
        citizenship = result["citizenshipLabel"]["value"]
        if country == "Ukraine":
            citizenship = "Ukraininan"
            
        language = result["languageLabel"]["value"]
        
        count_languages.append(result["languageCode"]["value"])

        if country not in countries_langs:
            countries_langs[country] = {
                "country_id": result["countryCode"]["value"],
                "citizenships": citizenship,
                "languages": [language],
                "languages_code": [result["languageCode"]["value"]],
            }
            cnt_countries += 1
        else:
            if result["citizenshipLabel"]["value"] and countries_langs[country]["citizenships"] and result["languageLabel"]["value"] in countries_langs[country]["languages"] and result["languageCode"]["value"] in countries_langs[country]["languages_code"]:
                continue
            else:
                #countries_langs[country]["citizenships"].append(citizenship)
                countries_langs[country]["languages"].append(language)
                countries_langs[country]["languages_code"].append(result["languageCode"]["value"])

        cnt += 1

    except Exception as e:
        print("#" * 10)
        print("language code not found", result["languageLabel"]["value"], result["languageCode"]["value"])
        continue

print("Total languages found: ", cnt)
print("Total unique languages found: ", len(count_languages))

for country in countries_langs:
    # countries_langs[country]["citizenships"] = list(
    #     countries_langs[country]["citizenships"]
    # )
    countries_langs[country]["languages"] = list(countries_langs[country]["languages"])
    countries_langs[country]["languages_code"] = list(
        countries_langs[country]["languages_code"]
    )

    if len(countries_langs[country]["citizenships"]) > 1:
        print(country, countries_langs[country]["citizenships"])


print(countries_langs)
print("Total countries: ", len(countries_langs))

with open("data/countries.json", "w", encoding="utf-8") as f:
    json.dump(countries_langs, f, indent=4)
