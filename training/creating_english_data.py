from googletrans import Translator
import pandas as pd

# Creating Translator Object
translator = Translator()

main = pd.read_csv("../data/marathi_data.csv")
print(main.head())

data = pd.DataFrame({'marathi':main['comment'],'rating':main['rating']})
data['english']=data['marathi'].apply(translator.translate,target_language='en').apply(getattr, args=('text',))
data.to_csv('../data/marathi_data_translated.csv')

## Old method for testing
##import goslate
##gs = goslate.Goslate()
##eng = []
##for i in range(len(data)):
##    eng_txt = gs.translate(data['marathi'][i],'en')
##    if(i % 100 == 0):
##        print(eng_txt)
##    eng.append(eng_txt)


