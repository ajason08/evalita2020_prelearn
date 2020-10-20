class datamining_config(object):
    ##data
    train = './pre_learn/crossdomain_splits/datamining_out.tsv'
    dev = './pre_learn/crossdomain_splits/datamining_only.tsv'
    test = './pre_learn/test_splits/datamining_only_test.tsv'

    ##model
    pretrained_model = 'dbmdz/bert-base-italian-uncased'
    best_model = './pre_learn/cross_domain/datamining/results_with_concepts_paragraph_bert_small_data_400steps/'

    ##training
    epochs = 5 


    ##output
    output_dir = './pre_learn/cross_domain/datamining/results_with_concepts_paragraph_bert_small_data_400steps'
    log_dir = './pre_learn/cross_domain/datamining/results_with_concepts_paragraph_bert_small_data_400steps/logs/'
    submission_file = './pre_learn/cross_domain/datamining/data_mining.csv'


class geometry_config(object):
    ##data
    train = './pre_learn/crossdomain_splits/geometry_out.tsv'
    dev = './pre_learn/crossdomain_splits/geometry_only.tsv'
    test = './pre_learn/test_splits/geometry_only_test.tsv'

    ##model
    pretrained_model = 'dbmdz/bert-base-italian-uncased'
    best_model = './pre_learn/cross_domain/geometry/results_with_concepts_paragraph_bert_small_data_400steps'
    ##training
    epochs = 5


    ##output
    output_dir = './pre_learn/cross_domain/geometry/results_with_concepts_paragraph_bert_small_data_400steps'
    log_dir = './pre_learn/cross_domain/geometry/results_with_concepts_paragraph_bert_small_data_400steps/logs/'
    submission_file = './pre_learn/cross_domain/geometry/geometry.csv'


class physics_config(object):
    ##data
    train = './pre_learn/crossdomain_splits/physics_out.tsv'
    dev = './pre_learn/crossdomain_splits/physics_only.tsv'
    test = './pre_learn/test_splits/physics_only_test.tsv'

    ##model
    pretrained_model = 'dbmdz/bert-base-italian-uncased'
    best_model = './pre_learn/cross_domain/physics/results_with_concepts_paragraph_bert_small_data_400steps/'
    ##training
    epochs = 5


    ##output
    output_dir = './pre_learn/cross_domain/physics/results_with_concepts_paragraph_bert_small_data_400steps'
    log_dir = './pre_learn/cross_domain/physics/results_with_concepts_paragraph_bert_small_data_400steps/logs/'
    submission_file = './pre_learn/cross_domain/physics.csv'


class precalculus_config(object):
    ##data
    train = './pre_learn/crossdomain_splits/precalculus_out.tsv'
    dev = './pre_learn/crossdomain_splits/precalculus_only.tsv'
    test = './pre_learn/test_splits/precalculus_only_test.tsv'

    ##model
    pretrained_model = 'dbmdz/bert-base-italian-uncased'
    best_model = './pre_learn/cross_domain/precalculus/results_with_concepts_paragraph_bert_small_data_400steps/'
    ##training
    epochs = 5


    ##output
    output_dir = './pre_learn/cross_domain/precalculus/results_with_concepts_paragraph_bert_small_data_400steps'
    log_dir = './pre_learn/cross_domain/precalculus/results_with_concepts_paragraph_bert_small_data_400steps/logs/'
    submission_file = './pre_learn/cross_domain/precalculus.csv'
