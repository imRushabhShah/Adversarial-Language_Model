
import nltk
from nltk.corpus import wordnet
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
import random
MAX_SEQUENCE_LENGTH = 5000



def get_tokens(sent):
    """
    return words tokens
    """
    return nltk.word_tokenize(sent)
    
def get_synonymes(word):
  
  '''
  using sysset from wordnet to get all possible synonyms
  '''
  
  def get_word(word_with_type):
    #get_synonymes('weak') -> returns all synonyms,all synonyms in dict with type of word
    """ 
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

def predict_sentence(model,sent):
    try_vector = tokenizer.texts_to_sequences([sent])
    try_vector = pad_sequences([try_vector[0]], 
                     maxlen=MAX_SEQUENCE_LENGTH, 
                     padding='pre', 
                     truncating='pre')
    val = model.predict(try_vector)
    return val

def attack(model,dummy,pertub=1, printSwaps = False):    
    dummy_temp = nltk.word_tokenize(dummy[:MAX_SEQUENCE_LENGTH])
    done = False
    unsuccessfullSwaps = 0
    for i in range(int(len(dummy_temp)*pertub)):
        if done:
            break
        itter = 0
        while itter<10000:
            itter+=1
            v = random.randint(0,len(dummy_temp)-1)
            syns,_ = get_synonymes(dummy_temp[v])
            if len(syns)>0:
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
#                 print("got changes in ",i,val)
                done = True
                break
    #         best_candidate = max(candidates.iteritems(), key=operator.itemgetter(1))[0]
        best_candidate = max(candidates, key=candidates.get)
        if word == best_candidate:
            unsuccessfullSwaps+=1
#             if printSwaps:
#                 print("did not swap")
#         else:
#             if printSwaps:
#                 print(word,best_candidate)
        dummy_temp[v] = best_candidate
    if not done:
#         print("sorry")
        return 0
    print("swaps done ",i+1-unsuccessfullSwaps,"total words",len(dummy_temp),"unsucessfull swaps try",unsuccessfullSwaps)
    return i+1-unsuccessfullSwaps