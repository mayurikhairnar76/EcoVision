"""Central configuration: image size, thresholds, disposal info, Eco Score and Marathi text."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "model" / "ecovision_cnn.keras"
CLASSES_PATH = ROOT / "model" / "class_names.json"
METRICS_PATH = ROOT / "model" / "metrics.json"

IMG_SIZE = (128, 128)
CONF_THRESHOLD = 0.60          # below this we refuse to give a recommendation
ALLOWED_EXT = {".jpg", ".jpeg", ".png"}

# Eco Score: a simple, documented HEURISTIC (not a scientific measurement).
# Higher = easier to recycle / lower end-of-life burden when segregated correctly.
#   Glass 90 : infinitely recyclable without quality loss
#   Metal 90 : high recycling rate, large energy saving vs. new metal
#   Paper 85 : widely recycled, but fibre degrades after several cycles
#   Organic 80: compostable, but only if kept out of mixed waste
#   Plastic 60: many types, often downcycled, high persistence if littered
WASTE_INFO = {
    "plastic": dict(bin="Dry Waste / Plastic Recycling", bin_mr="सुका कचरा / प्लास्टिक पुनर्वापर",
                    recyclable=True, eco=60, impact="High", impact_mr="जास्त",
                    tip="Rinse and separate plastic containers before recycling.",
                    tip_mr="पुनर्वापरापूर्वी प्लास्टिकचे डबे धुवून वेगळे करा.",
                    info="Most rigid plastics (bottles, tubs) are recyclable; thin films often are not.",
                    info_mr="बाटल्या, डबे बहुतांशी पुनर्वापरयोग्य असतात; पातळ पिशव्या सहसा नसतात."),
    "paper": dict(bin="Paper Recycling (Dry Waste)", bin_mr="कागद पुनर्वापर (सुका कचरा)",
                  recyclable=True, eco=85, impact="Low", impact_mr="कमी",
                  tip="Keep paper dry and clean; greasy or wet paper cannot be recycled.",
                  tip_mr="कागद कोरडा व स्वच्छ ठेवा; तेलकट किंवा ओला कागद पुनर्वापरास अयोग्य.",
                  info="Paper fibres can be recycled several times.",
                  info_mr="कागदाचे तंतू अनेकदा पुनर्वापरता येतात."),
    "glass": dict(bin="Glass Recycling (Dry Waste)", bin_mr="काच पुनर्वापर (सुका कचरा)",
                  recyclable=True, eco=90, impact="Low", impact_mr="कमी",
                  tip="Wrap broken glass safely and label it before disposal.",
                  tip_mr="फुटलेली काच सुरक्षितपणे गुंडाळून त्यावर लिहून द्या.",
                  info="Glass can be recycled repeatedly without losing quality.",
                  info_mr="काच गुणवत्ता न गमावता वारंवार पुनर्वापरता येते."),
    "metal": dict(bin="Metal Recycling (Dry Waste)", bin_mr="धातू पुनर्वापर (सुका कचरा)",
                  recyclable=True, eco=90, impact="Low", impact_mr="कमी",
                  tip="Empty and rinse cans; metal scrap can be sold to recyclers.",
                  tip_mr="डबे रिकामे करून धुवा; धातूचा भंगार पुनर्वापरकर्त्यांना विका.",
                  info="Recycling metal saves a large amount of energy vs. mining new ore.",
                  info_mr="धातूच्या पुनर्वापराने नवीन खनिज उत्खननापेक्षा खूप ऊर्जा वाचते."),
    "organic": dict(bin="Wet Waste / Compost", bin_mr="ओला कचरा / कंपोस्ट",
                    recyclable=False, eco=80, impact="Medium", impact_mr="मध्यम",
                    tip="Compost food scraps at home or use the wet-waste bin.",
                    tip_mr="अन्नाचा कचरा घरी कंपोस्ट करा किंवा ओल्या कचऱ्याच्या पेटीत टाका.",
                    info="Not recyclable, but compostable into manure.",
                    info_mr="पुनर्वापरयोग्य नाही, पण कंपोस्ट करून खत बनवता येते."),
}
NAME_MR = {"plastic": "प्लास्टिक", "paper": "कागद", "glass": "काच",
           "metal": "धातू", "organic": "सेंद्रिय कचरा"}

TEXT = {
    "English": dict(title="EcoVision", sub="Smart Waste Detection and Segregation using CNN",
        upload="Upload a waste image (JPG/PNG)", predict="Predict", reset="Reset",
        waste="Waste Type", conf="Confidence", bin="Recommended Bin", eco="Eco Score",
        recyclable="Recyclable", yes="Yes", no="No (compostable)", impact="Environmental Impact",
        info="Recycling Information", tip="Eco Tip", xai="Explainable AI (Grad-CAM)",
        xai_note="The highlighted region shows the part of the image that most influenced the CNN prediction.",
        orig="Original", heat="Grad-CAM overlay",
        low="The model is not sufficiently confident about this image. Please upload a clearer image.",
        noimg="Please upload an image first.", dash="Dashboard", cats="Waste categories",
        acc="Test accuracy", preds="Predictions this session", last="Last detected",
        rec_cnt="Recyclable / Non-recyclable", disclaimer="Eco Score is a simple heuristic, not a scientific measurement."),
    "मराठी": dict(title="EcoVision", sub="CNN द्वारे स्मार्ट कचरा ओळख व वर्गीकरण",
        upload="कचऱ्याचे चित्र अपलोड करा (JPG/PNG)", predict="अंदाज लावा", reset="रीसेट",
        waste="कचऱ्याचा प्रकार", conf="विश्वास", bin="सुचवलेली कचरापेटी", eco="इको स्कोअर",
        recyclable="पुनर्वापरयोग्य", yes="होय", no="नाही (कंपोस्ट करता येते)", impact="पर्यावरणीय परिणाम",
        info="पुनर्वापर माहिती", tip="पर्यावरण टिप", xai="स्पष्टीकरणक्षम AI (Grad-CAM)",
        xai_note="हायलाइट केलेला भाग CNN च्या अंदाजावर सर्वाधिक प्रभाव टाकणारा भाग दर्शवतो.",
        orig="मूळ चित्र", heat="Grad-CAM आच्छादन",
        low="मॉडेलला या चित्राबद्दल पुरेशी खात्री नाही. कृपया अधिक स्पष्ट चित्र अपलोड करा.",
        noimg="कृपया आधी चित्र अपलोड करा.", dash="डॅशबोर्ड", cats="कचरा प्रकार",
        acc="चाचणी अचूकता", preds="या सत्रातील अंदाज", last="शेवटचा ओळखलेला",
        rec_cnt="पुनर्वापरयोग्य / अयोग्य", disclaimer="इको स्कोअर हा साधा अंदाजित निकष आहे, वैज्ञानिक मोजमाप नाही."),
}
