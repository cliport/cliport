from cliport.agents.transporter import OriginalTransporterAgent
from cliport.agents.transporter import ClipUNetTransporterAgent
from cliport.agents.transporter import TwoStreamClipWithoutSkipsTransporterAgent
from cliport.agents.transporter import TwoStreamRN50BertUNetTransporterAgent
from cliport.agents.transporter import TwoStreamClipUNetTransporterAgent

from cliport.agents.transporter_lang_goal import TwoStreamClipLingUNetTransporterAgent
from cliport.agents.transporter_lang_goal import TwoStreamRN50BertLingUNetTransporterAgent
from cliport.agents.transporter_lang_goal import TwoStreamUntrainedRN50BertLingUNetTransporterAgent
from cliport.agents.transporter_lang_goal import OriginalTransporterLangFusionAgent
from cliport.agents.transporter_lang_goal import ClipLingUNetTransporterAgent
from cliport.agents.transporter_lang_goal import TwoStreamRN50BertLingUNetLatTransporterAgent

from cliport.agents.transporter_image_goal import ImageGoalTransporterAgent

from cliport.agents.transporter import TwoStreamClipUNetLatTransporterAgent
from cliport.agents.transporter_lang_goal import TwoStreamClipLingUNetLatTransporterAgent
from cliport.agents.transporter_lang_goal import TwoStreamClipFilmLingUNetLatTransporterAgent


names = {
         ################################
         ### CLIPort ###
         'cliport': TwoStreamClipLingUNetLatTransporterAgent,
         'two_stream_clip_lingunet_lat_transporter': TwoStreamClipLingUNetLatTransporterAgent,

         ################################
         ### Two-Stream Architectures ###
         # CLIPort without language
         'two_stream_clip_unet_lat_transporter': TwoStreamClipUNetLatTransporterAgent,

         # CLIPort without lateral connections
         'two_stream_clip_lingunet_transporter': TwoStreamClipLingUNetTransporterAgent,

         # CLIPort without language and lateral connections
         'two_stream_clip_unet_transporter': TwoStreamClipUNetTransporterAgent,

         # CLIPort without language, lateral, or skip connections
         'two_stream_clip_woskip_transporter': TwoStreamClipWithoutSkipsTransporterAgent,

         # RN50-BERT
         'two_stream_full_rn50_bert_lingunet_lat_transporter': TwoStreamRN50BertLingUNetLatTransporterAgent,

         # RN50-BERT without language
         'two_stream_full_rn50_bert_unet_transporter': TwoStreamRN50BertUNetTransporterAgent,

         # RN50-BERT without lateral connections
         'two_stream_full_rn50_bert_lingunet_transporter': TwoStreamRN50BertLingUNetTransporterAgent,

         # Untrained RN50-BERT (similar to untrained CLIP)
         'two_stream_full_untrained_rn50_bert_lingunet_transporter': TwoStreamUntrainedRN50BertLingUNetTransporterAgent,

         ###################################
         ### Single-Stream Architectures ###
         # Transporter-only
         'transporter': OriginalTransporterAgent,

         # CLIP-only without language
         'clip_unet_transporter': ClipUNetTransporterAgent,

         # CLIP-only
         'clip_lingunet_transporter': ClipLingUNetTransporterAgent,

         # Transporter with language (at bottleneck)
         'transporter_lang': OriginalTransporterLangFusionAgent,

         # Image-Goal Transporter
         'image_goal_transporter': ImageGoalTransporterAgent,

         ##############################################
         ### New variants NOT reported in the paper ###

         # CLIPort with FiLM language fusion
         'two_stream_clip_film_lingunet_lat_transporter': TwoStreamClipFilmLingUNetLatTransporterAgent,
         }