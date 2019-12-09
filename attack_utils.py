import nltk
from nltk.corpus import wordnet
import random
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')
def get_tokens(sent):
    """
    return words tokens
    """
    return nltk.word_tokenize(sent)
    
def get_synonymes(word):
  # using sysset from wordnet to get all possible synonyms  
  def get_word(word_with_type):
    """ 
    #get_synonymes('weak') -> returns all synonyms,all synonyms in dict with type of word
    the function takes a string like "dog.n.01" and return dog,n so we know what
    type of word we have like is it verb for v adj for a and n for noun
    """
    word_new = word_with_type.split('.')[0]
    word_new = word_new.replace('_'," ")
    return (word_new,word_with_type.split('.')[1])
  
  synsets = wordnet.synsets(word)
  syn = [x.name() for x in synsets]
  for x in synsets:
    syn+=list(i.name() for i in x.hypernyms())
  syn = set(get_word(s) for s in syn)
  sss = set()
  for s in syn:
    a,b = s
    if a == word:
      continue
    sss.add(a)
  syn_dict = {}
  for w,t in syn:
    if t not in syn_dict:
      syn_dict[t]=set()
    if w == word:
      continue
    syn_dict[t].add(w)
  return sss,syn_dict


def evaluate_ourfakes(model, Fake_texts):
    '''
    evaluate the model, both training and testing errors are reported
    '''
    missclasified_fakes = []
    correct_clasified_fakes = []
    def predict(X):
        return np.rint(model.predict(X))
    # training error
    y_predict = predict(Fake_texts)
    y = np.zeros(len(y_predict))
    for i in range(len(y_predict)):
        missclasified_fakes.append(Fake_texts[i]) if y_predict[i] == 1 else correct_clasified_fakes.append(Fake_texts[i])
    acc = accuracy_score(y,y_predict) #model.evaluate(Fake_texts,y)
    print("total missclassified fakes",len(missclasified_fakes),"total correctly classified fakes",len(correct_clasified_fakes))
    return correct_clasified_fakes

def predict_sentence(model,sent):
    try_vector = model.tokenizer.texts_to_sequences([sent])
    try_vector = pad_sequences([try_vector[0]], 
                     maxlen=MAX_SEQUENCE_LENGTH, 
                     padding='pre', 
                     truncating='pre')
    val = model.model.predict(try_vector)
    return val


def attack(model,dummy,pertub=1, printSwaps = False): 
    stopwords_list = set(stopwords.words('english'))   
    dummy_temp = nltk.word_tokenize(dummy[:MAX_SEQUENCE_LENGTH])
    second_dummy = nltk.word_tokenize(dummy[:MAX_SEQUENCE_LENGTH])
    success_index=[]
    done = False
    swaps = 0
    for i in range(int(len(dummy_temp)*pertub)):
        if done:
            break
        itter = 0
        while itter<1000:
            itter+=1
            v = random.randint(0,len(dummy_temp)-1)
            if dummy_temp[v] in stopwords_list or v in success_index:
              continue
            syns,_ = get_synonymes(dummy_temp[v])
            if len(syns)>0 :
                break
        if itter == 10000:
            print("didnt catch any synonymes")
            continue
        candidates = {}
        # candidate is a scored candidate dictionary storing all the 
        # synonymes with prediction Score
        word = dummy_temp[v]
        sent = " ".join(dummy_temp)
        candidates[word]=predict_sentence(model,sent)
        for s in syns:
            dummy_temp[v] = s
            sent = (" ".join(dummy_temp))
            val = predict_sentence(model,sent)
            candidates[s] = val
            if val>=0.5:
                success_index.append(v)
                done = True
                break
        best_candidate = max(candidates, key=candidates.get)
        if not word == best_candidate:
            success_index.append(v)
            swaps+=1
        dummy_temp[v] = best_candidate
    if not done:
        return 0,None
    for ind in success_index:
        dummy_temp[ind] = '<attack>'+dummy_temp[ind]+'</attack><original>'+second_dummy[ind]+"</original>"
    return swaps," ".join(dummy_temp)