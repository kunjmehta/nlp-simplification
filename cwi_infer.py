from cwi_train import Data, Instance, auc, FeatureExtractor
from keras.models import load_model
import csv

database_path="DomainSpecific.xlsx"

def to_tsv(inpath, outpath):
    print(inpath, '\n',outpath)
    out_file =  open(outpath, 'wt', newline = '')
    tsv_writer = csv.writer(out_file, delimiter='\t')

    with open(inpath, 'r') as in_file:
        line_list = in_file.read().split('\n')
        print(line_list)
        line_index = 1
        for line in line_list:
            print(line, '   ', len(line))
            start_index = 0
            for i in range(len(line)):
                if line[i] == ' ':
                    # print(line[start_index:i], "  ", start_index, "   ", i)
                    tsv_writer.writerow(['HIT_id', line, start_index, i - 1, line[start_index:i], '10', '10'])
                    start_index = i + 1
                    # print(line[start_index:], "  ", start_index, "   ", )
            tsv_writer.writerow(['HIT_id', line, start_index, len(line) - 1, line[start_index:], '10', '10'])

            # tsv_writer.writerow()


def word_pred_map(pred_test, data_input_para):
    word_to_pred = {}

    for i in range(len(pred_test)):
        word_to_pred[data_input_para.instances[i].target_chars] = pred_test[i]

    return word_to_pred
    # print(word_to_pred)

def extract_complex_words(neural_network_model_path, embeddings_path, tsvfile):
    fe = FeatureExtractor(embedding_model_path=embeddings_path)
    model = load_model(neural_network_model_path, custom_objects={'auc': auc})
    input_para = [tsvfile]
    data_input_para = Data(input_para, is_test=True)
    input_embeddings = fe.predict_average_embeddings(data_input_para.instances)
    input_pred = model.predict(input_embeddings)

    print(input_pred)

    mapping = word_pred_map(input_pred, data_input_para)
    # print(mapping)
    return  mapping


def read_custom_db():

    df = pd.read_excel (database_path)

    synonym_dict = dict(zip(df['Complex Word'], df['simple word']))
    synonym_dict = {k: synonym_dict[k] for k in synonym_dict if type(synonym_dict[k]) is str}
    print(synonym_dict)


    phrase_dict = dict(zip(df['Complex Word'], df['Simple phrase']))
    phrase_dict = {k: phrase_dict[k] for k in phrase_dict if type(phrase_dict[k]) is str}
    print(phrase_dict)


