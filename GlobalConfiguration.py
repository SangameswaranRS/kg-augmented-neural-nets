# ------------ Global Configuration for the Project -----------#

class DKRL_CONFIG:
    DKRL_HOME = '/home/sangameswaran/DKRL'

class PROJECT_CONFIG:
    PROJECT_HOME = '/home/sangameswaran/FYP/code/build'
    PROJECT_HOME_TEST = '/home/sangameswaran/FYP/code/test'

class ENTITY_RETRIEVAL_CONFIG:
    ATTENTION_OUTPUT_RELATIVE_PATH = '/entity_attention.csv'
    def get_attention_file_path(self, build_status= 0):
        if build_status == 0:
            return PROJECT_CONFIG.PROJECT_HOME_TEST + self.ATTENTION_OUTPUT_RELATIVE_PATH
        elif build_status == 1:
            return PROJECT_CONFIG.PROJECT_HOME + self.ATTENTION_OUTPUT_RELATIVE_PATH

class RELATION_RETRIEVAL_CONFIG:
    ATTENTION_OUTPUT_RELATIVE_PATH = '/relation_attention.csv'
    def get_attention_file_path(self, build_status= 0):
        if build_status == 0:
            return PROJECT_CONFIG.PROJECT_HOME_TEST + self.ATTENTION_OUTPUT_RELATIVE_PATH
        elif build_status == 1:
            return PROJECT_CONFIG.PROJECT_HOME + self.ATTENTION_OUTPUT_RELATIVE_PATH

