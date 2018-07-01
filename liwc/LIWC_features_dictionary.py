import xlrd
keys_to_indexes = {
    "I": [5],
    "We": [6],
    "They": [9],
    
    #Spoken categories
    "Non-fluencies": [103],
    
    #Psychology
    "PosEmotion": [39,40,41],
    "NegEmotion": [42,43,44,45],
    "Anxiety": [46],
    "Anger": [47,48],
    "Sadness": [49],
    "Cognitive": [50,51,52,53,54],
    "Inhibition": [61],
    "Perceptual": [64,65],
    "Relativity": [79,80,81,82,83]
#     "Work": [89,90,91],
#     "Swear": [26],
#     "Articles": [11],
#     "Verbs": [12,13,14],
#     "Adverbs": [20],
#     "Prepositions": [21],
#     "Conjunctions": [22],
#     "Negations": [23],
#     "Quantifiers": [24],
#     "Numbers": [25]
}

book = xlrd.open_workbook("liwc/LIWC.xlsx")
sh = book.sheet_by_index(0)

def get_dictionary() -> dict:
    return {
        key: [
            sh.cell_value(rowx=rowi, colx=coli)
            for rowi in range(3, sh.nrows)
            for coli in value
            if(sh.cell_value(rowx=rowi, colx=coli) != "")
            ] 
        for key, value in keys_to_indexes.items()
    }






