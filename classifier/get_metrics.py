# path_to_correct_sentences = "/home/rmarefat/projects/PSC/Nevise_Cleaned/DATA/TXTs_Sentences/EXAMPLE_CORRECT.txt"
# path_to_predicted_sentences = "/home/rmarefat/projects/PSC/Nevise_Cleaned/DATA/TXTs_Sentences/EXAMPLE_PREDICTED.txt"
# path_to_manipulation_GT = "/home/rmarefat/projects/PSC/Nevise_Cleaned/DATA/TXTs_Sentences/EXAMPLE_CORRECT_MANIPULATION_GT.txt"
path_to_correct_sentences = "/home/sekhavat/projects/Nevise_Cleaned/DATA/infer/nevise-news-451_normal_corrects.txt"
path_to_predicted_sentences = "/home/sekhavat/projects/Nevise_Cleaned/DATA/infer/nevise-news-451_wrongs_PREDICTED.txt"
path_to_manipulation_GT = "/home/sekhavat/projects/Nevise_Cleaned/DATA/infer/nevise-news-451_MANIPULATION_GT.txt"

number_of_correctly_detected_wrong_words = 0
number_of_correctly_corrected_words = 0
number_of_wrongly_detected_correct_words = 0
number_of_all_wrong_words = 0
number_of_all_correct_words = 0

with open(path_to_correct_sentences) as h:
    correct_sentences = [f.replace("\n", "") for f in h.readlines()]

with open(path_to_predicted_sentences) as h:
    predicted_sentences = [f.replace("\n", "") for f in h.readlines()]

with open(path_to_manipulation_GT) as h:
    manipulation_gt = [f.replace("\n", "") for f in h.readlines()]

for cs, ps, mgt in zip(correct_sentences, predicted_sentences, manipulation_gt):
    cs, ps, mgts = cs.strip().split(" "), ps.strip(" ").split(), mgt.strip().split(" ")

    if len(cs) != len(ps):
        continue
    for cw, pw, mgt in zip(cs, ps, mgts):
        
        if cw == pw and mgt == "1":
            number_of_correctly_corrected_words += 1
            number_of_correctly_detected_wrong_words += 1

        elif not(cw == pw) and mgt == "0":

            number_of_wrongly_detected_correct_words += 1

        if mgt == "0":
            number_of_all_correct_words += 1
        elif mgt == "1":
            number_of_all_wrong_words += 1
            
print("number_of_all_correct_words", number_of_all_correct_words)
print("number_of_all_wrong_words", number_of_all_wrong_words)
print("number_of_correctly_corrected_words", number_of_correctly_corrected_words)
print("number_of_correctly_detected_wrong_words", number_of_correctly_detected_wrong_words)
print("number_of_wrongly_detected_correct_words", number_of_wrongly_detected_correct_words)


