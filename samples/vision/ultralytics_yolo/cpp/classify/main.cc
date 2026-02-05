/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

Copyright (c) 2024-2025, WuChao && MaChao D-Robotics.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

// 注意: 此程序在RDK板端运行
// Attention: This program runs on RDK board.

// ============================================================================
// Configuration Parameters
// ============================================================================

// D-Robotics *.bin 模型路径
// Path to D-Robotics *.bin model
#define MODEL_PATH "source/reference_bin_models/cls/yolo11n_cls_bayese_224x224_nv12.bin"

// 测试图片路径
// Path to test image
#define TEST_IMG_PATH "../../../../datasets/imagenet/asset/zebra_cls.jpg"

// 前处理方式: 0=Resize, 1=LetterBox
// Preprocessing method: 0=Resize, 1=LetterBox
#define RESIZE_TYPE 0
#define LETTERBOX_TYPE 1
#define PREPROCESS_TYPE LETTERBOX_TYPE

// Top K 结果数量
// Number of top K results to display
#define TOP_K 5

// ============================================================================
// Includes
// ============================================================================

#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <numeric>

// OpenCV
#include <opencv2/opencv.hpp>

// RDK BPU libDNN API
#include "dnn/hb_dnn.h"
#include "dnn/hb_dnn_ext.h"
#include "dnn/plugin/hb_dnn_layer.h"
#include "dnn/plugin/hb_dnn_plugin.h"
#include "dnn/hb_sys.h"

// ============================================================================
// Macros
// ============================================================================

#define CHECK_SUCCESS(value, errmsg)                                         \
    do {                                                                     \
        auto ret_code = value;                                               \
        if (ret_code != 0) {                                                 \
            std::cerr << "\033[1;31m[ERROR]\033[0m " << __FILE__ << ":"     \
                      << __LINE__ << " " << errmsg                           \
                      << ", error code: " << ret_code << std::endl;          \
            return ret_code;                                                 \
        }                                                                    \
    } while (0)

#define LOG_INFO(msg) \
    std::cout << "\033[1;32m[INFO]\033[0m " << msg << std::endl

#define LOG_WARN(msg) \
    std::cout << "\033[1;33m[WARN]\033[0m " << msg << std::endl

#define LOG_ERROR(msg) \
    std::cerr << "\033[1;31m[ERROR]\033[0m " << msg << std::endl

#define LOG_TIME(msg, duration) \
    std::cout << "\033[1;31m" << msg << " = " << std::fixed            \
              << std::setprecision(2) << (duration) << " ms\033[0m"    \
              << std::endl

// ============================================================================
// ImageNet 1000 Classes (abbreviated for brevity, full list in Python demo)
// ============================================================================

const std::vector<std::string> IMAGENET_CLASSES = {
    "tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark",
    "electric ray", "stingray", "cock", "hen", "ostrich",
    "brambling", "goldfinch", "house finch", "junco", "indigo bunting",
    "robin", "bulbul", "jay", "magpie", "chickadee",
    "water ouzel", "kite", "bald eagle", "vulture", "great grey owl",
    "European fire salamander", "common newt", "eft", "spotted salamander", "axolotl",
    "bullfrog", "tree frog", "tailed frog", "loggerhead", "leatherback turtle",
    "mud turtle", "terrapin", "box turtle", "banded gecko", "common iguana",
    "American chameleon", "whiptail", "agama", "frilled lizard", "alligator lizard",
    "Gila monster", "green lizard", "African chameleon", "Komodo dragon", "African crocodile",
    "American alligator", "triceratops", "thunder snake", "ringneck snake", "hognose snake",
    "green snake", "king snake", "garter snake", "water snake", "vine snake",
    "night snake", "boa constrictor", "rock python", "Indian cobra", "green mamba",
    "sea snake", "horned viper", "diamondback", "sidewinder", "trilobite",
    "harvestman", "scorpion", "black and gold garden spider", "barn spider", "garden spider",
    "black widow", "tarantula", "wolf spider", "tick", "centipede",
    "black grouse", "ptarmigan", "ruffed grouse", "prairie chicken", "peacock",
    "quail", "partridge", "African grey", "macaw", "sulphur-crested cockatoo",
    "lorikeet", "coucal", "bee eater", "hornbill", "hummingbird",
    "jacamar", "toucan", "drake", "red-breasted merganser", "goose",
    "black swan", "tusker", "echidna", "platypus", "wallaby",
    "koala", "wombat", "jellyfish", "sea anemone", "brain coral",
    "flatworm", "nematode", "conch", "snail", "slug",
    "sea slug", "chiton", "chambered nautilus", "Dungeness crab", "rock crab",
    "fiddler crab", "king crab", "American lobster", "spiny lobster", "crayfish",
    "hermit crab", "isopod", "white stork", "black stork", "spoonbill",
    "flamingo", "little blue heron", "American egret", "bittern", "crane",
    "limpkin", "European gallinule", "American coot", "bustard", "ruddy turnstone",
    "red-backed sandpiper", "redshank", "dowitcher", "oystercatcher", "pelican",
    "king penguin", "albatross", "grey whale", "killer whale", "dugong",
    "sea lion", "Chihuahua", "Japanese spaniel", "Maltese dog", "Pekinese",
    "Shih-Tzu", "Blenheim spaniel", "papillon", "toy terrier", "Rhodesian ridgeback",
    "Afghan hound", "basset", "beagle", "bloodhound", "bluetick",
    "black-and-tan coonhound", "Walker hound", "English foxhound", "redbone", "borzoi",
    "Irish wolfhound", "Italian greyhound", "whippet", "Ibizan hound", "Norwegian elkhound",
    "otterhound", "Saluki", "Scottish deerhound", "Weimaraner", "Staffordshire bullterrier",
    "American Staffordshire terrier", "Bedlington terrier", "Border terrier", "Kerry blue terrier", "Irish terrier",
    "Norfolk terrier", "Norwich terrier", "Yorkshire terrier", "wire-haired fox terrier", "Lakeland terrier",
    "Sealyham terrier", "Airedale", "cairn", "Australian terrier", "Dandie Dinmont",
    "Boston bull", "miniature schnauzer", "giant schnauzer", "standard schnauzer", "Scotch terrier",
    "Tibetan terrier", "silky terrier", "soft-coated wheaten terrier", "West Highland white terrier", "Lhasa",
    "flat-coated retriever", "curly-coated retriever", "golden retriever", "Labrador retriever", "Chesapeake Bay retriever",
    "German short-haired pointer", "vizsla", "English setter", "Irish setter", "Gordon setter",
    "Brittany spaniel", "clumber", "English springer", "Welsh springer spaniel", "cocker spaniel",
    "Sussex spaniel", "Irish water spaniel", "kuvasz", "schipperke", "groenendael",
    "malinois", "briard", "kelpie", "komondor", "Old English sheepdog",
    "Shetland sheepdog", "collie", "Border collie", "Bouvier des Flandres", "Rottweiler",
    "German shepherd", "Doberman", "miniature pinscher", "Greater Swiss Mountain dog", "Bernese mountain dog",
    "Appenzeller", "EntleBucher", "boxer", "bull mastiff", "Tibetan mastiff",
    "French bulldog", "Great Dane", "Saint Bernard", "Eskimo dog", "malamute",
    "Siberian husky", "dalmatian", "affenpinscher", "basenji", "pug",
    "Leonberg", "Newfoundland", "Great Pyrenees", "Samoyed", "Pomeranian",
    "chow", "keeshond", "Brabancon griffon", "Pembroke", "Cardigan",
    "toy poodle", "miniature poodle", "standard poodle", "Mexican hairless", "timber wolf",
    "white wolf", "red wolf", "coyote", "dingo", "dhole",
    "African hunting dog", "hyena", "red fox", "kit fox", "Arctic fox",
    "grey fox", "tabby", "tiger cat", "Persian cat", "Siamese cat",
    "Egyptian cat", "cougar", "lynx", "leopard", "snow leopard",
    "jaguar", "lion", "tiger", "cheetah", "brown bear",
    "American black bear", "ice bear", "sloth bear", "mongoose", "meerkat",
    "tiger beetle", "ladybug", "ground beetle", "long-horned beetle", "leaf beetle",
    "dung beetle", "rhinoceros beetle", "weevil", "fly", "bee",
    "ant", "grasshopper", "cricket", "walking stick", "cockroach",
    "mantis", "cicada", "leafhopper", "lacewing", "dragonfly",
    "damselfly", "admiral", "ringlet", "monarch", "cabbage butterfly",
    "sulphur butterfly", "lycaenid", "starfish", "sea urchin", "sea cucumber",
    "wood rabbit", "hare", "Angora", "hamster", "porcupine",
    "fox squirrel", "marmot", "beaver", "guinea pig", "sorrel",
    "zebra", "hog", "wild boar", "warthog", "hippopotamus",
    "ox", "water buffalo", "bison", "ram", "bighorn",
    "ibex", "hartebeest", "impala", "gazelle", "Arabian camel",
    "llama", "weasel", "mink", "polecat", "black-footed ferret",
    "otter", "skunk", "badger", "armadillo", "three-toed sloth",
    "orangutan", "gorilla", "chimpanzee", "gibbon", "siamang",
    "guenon", "patas", "baboon", "macaque", "langur",
    "colobus", "proboscis monkey", "marmoset", "capuchin", "howler monkey",
    "titi", "spider monkey", "squirrel monkey", "Madagascar cat", "indri",
    "Indian elephant", "African elephant", "lesser panda", "giant panda", "barracouta",
    "eel", "coho", "rock beauty", "anemone fish", "sturgeon",
    "gar", "lionfish", "puffer", "abacus", "abaya",
    "academic gown", "accordion", "acoustic guitar", "aircraft carrier", "airliner",
    "airship", "altar", "ambulance", "amphibian", "analog clock",
    "apiary", "apron", "ashcan", "assault rifle", "backpack",
    "bakery", "balance beam", "balloon", "ballpoint", "Band Aid",
    "banjo", "bannister", "barbell", "barber chair", "barbershop",
    "barn", "barometer", "barrel", "barrow", "baseball",
    "basketball", "bassinet", "bassoon", "bathing cap", "bath towel",
    "bathtub", "beach wagon", "beacon", "beaker", "bearskin",
    "beer bottle", "beer glass", "bell cote", "bib", "bicycle-built-for-two",
    "bikini", "binder", "binoculars", "birdhouse", "boathouse",
    "bobsled", "bolo tie", "bonnet", "bookcase", "bookshop",
    "bottlecap", "bow", "bow tie", "brass", "brassiere",
    "breakwater", "breastplate", "broom", "bucket", "buckle",
    "bulletproof vest", "bullet train", "butcher shop", "cab", "caldron",
    "candle", "cannon", "canoe", "can opener", "cardigan",
    "car mirror", "carousel", "carpenter's kit", "carton", "car wheel",
    "cash machine", "cassette", "cassette player", "castle", "catamaran",
    "CD player", "cello", "cellular telephone", "chain", "chainlink fence",
    "chain mail", "chain saw", "chest", "chiffonier", "chime",
    "china cabinet", "Christmas stocking", "church", "cinema", "cleaver",
    "cliff dwelling", "cloak", "clog", "cocktail shaker", "coffee mug",
    "coffeepot", "coil", "combination lock", "computer keyboard", "confectionery",
    "container ship", "convertible", "corkscrew", "cornet", "cowboy boot",
    "cowboy hat", "cradle", "crane2", "crash helmet", "crate",
    "crib", "Crock Pot", "croquet ball", "crutch", "cuirass",
    "dam", "desk", "desktop computer", "dial telephone", "diaper",
    "digital clock", "digital watch", "dining table", "dishrag", "dishwasher",
    "disk brake", "dock", "dogsled", "dome", "doormat",
    "drilling platform", "drum", "drumstick", "dumbbell", "Dutch oven",
    "electric fan", "electric guitar", "electric locomotive", "entertainment center", "envelope",
    "espresso maker", "face powder", "feather boa", "file", "fireboat",
    "fire engine", "fire screen", "flagpole", "flute", "folding chair",
    "football helmet", "forklift", "fountain", "fountain pen", "four-poster",
    "freight car", "French horn", "frying pan", "fur coat", "garbage truck",
    "gasmask", "gas pump", "goblet", "go-kart", "golf ball",
    "golfcart", "gondola", "gong", "gown", "grand piano",
    "greenhouse", "grille", "grocery store", "guillotine", "hair slide",
    "hair spray", "half track", "hammer", "hamper", "hand blower",
    "hand-held computer", "handkerchief", "hard disc", "harmonica", "harp",
    "harvester", "hatchet", "holster", "home theater", "honeycomb",
    "hook", "hoopskirt", "horizontal bar", "horse cart", "hourglass",
    "iPod", "iron", "jack-o'-lantern", "jean", "jeep",
    "jersey", "jigsaw puzzle", "jinrikisha", "joystick", "kimono",
    "knee pad", "knot", "lab coat", "ladle", "lampshade",
    "laptop", "lawn mower", "lens cap", "letter opener", "library",
    "lifeboat", "lighter", "limousine", "liner", "lipstick",
    "Loafer", "lotion", "loudspeaker", "loupe", "lumbermill",
    "magnetic compass", "mailbag", "mailbox", "maillot", "maillot",
    "manhole cover", "maraca", "marimba", "mask", "matchstick",
    "maypole", "maze", "measuring cup", "medicine chest", "megalith",
    "microphone", "microwave", "military uniform", "milk can", "minibus",
    "miniskirt", "minivan", "missile", "mitten", "mixing bowl",
    "mobile home", "Model T", "modem", "monastery", "monitor",
    "moped", "mortar", "mortarboard", "mosque", "mosquito net",
    "motor scooter", "mountain bike", "mountain tent", "mouse", "mousetrap",
    "moving van", "muzzle", "nail", "neck brace", "necklace",
    "nipple", "notebook", "obelisk", "oboe", "ocarina",
    "odometer", "oil filter", "organ", "oscilloscope", "overskirt",
    "oxcart", "oxygen mask", "packet", "paddle", "paddlewheel",
    "padlock", "paintbrush", "pajama", "palace", "panpipe",
    "paper towel", "parachute", "parallel bars", "park bench", "parking meter",
    "passenger car", "patio", "pay-phone", "pedestal", "pencil box",
    "pencil sharpener", "perfume", "Petri dish", "photocopier", "pick",
    "pickelhaube", "picket fence", "pickup", "pier", "piggy bank",
    "pill bottle", "pillow", "ping-pong ball", "pinwheel", "pirate",
    "pitcher", "plane", "planetarium", "plastic bag", "plate rack",
    "plow", "plunger", "Polaroid camera", "pole", "police van",
    "poncho", "pool table", "pop bottle", "pot", "potter's wheel",
    "power drill", "prayer rug", "printer", "prison", "projectile",
    "projector", "puck", "punching bag", "purse", "quill",
    "quilt", "racer", "racket", "radiator", "radio",
    "radio telescope", "rain barrel", "recreational vehicle", "reel", "reflex camera",
    "refrigerator", "remote control", "restaurant", "revolver", "rifle",
    "rocking chair", "rotisserie", "rubber eraser", "rugby ball", "rule",
    "running shoe", "safe", "safety pin", "saltshaker", "sandal",
    "sarong", "sax", "scabbard", "scale", "school bus",
    "schooner", "scoreboard", "screen", "screw", "screwdriver",
    "seat belt", "sewing machine", "shield", "shoe shop", "shoji",
    "shopping basket", "shopping cart", "shovel", "shower cap", "shower curtain",
    "ski", "ski mask", "sleeping bag", "slide rule", "sliding door",
    "slot", "snorkel", "snowmobile", "snowplow", "soap dispenser",
    "soccer ball", "sock", "solar dish", "sombrero", "soup bowl",
    "space bar", "space heater", "space shuttle", "spatula", "speedboat",
    "spider web", "spindle", "sports car", "spotlight", "stage",
    "steam locomotive", "steel arch bridge", "steel drum", "stethoscope", "stole",
    "stone wall", "stopwatch", "stove", "strainer", "streetcar",
    "stretcher", "studio couch", "stupa", "submarine", "suit",
    "sundial", "sunglass", "sunglasses", "sunscreen", "suspension bridge",
    "swab", "sweatshirt", "swimming trunks", "swing", "switch",
    "syringe", "table lamp", "tank", "tape player", "teapot",
    "teddy", "television", "tennis ball", "thatch", "theater curtain",
    "thimble", "thresher", "throne", "tile roof", "toaster",
    "tobacco shop", "toilet seat", "torch", "totem pole", "tow truck",
    "toyshop", "tractor", "trailer truck", "tray", "trench coat",
    "tricycle", "trimaran", "tripod", "triumphal arch", "trolleybus",
    "trombone", "tub", "turnstile", "typewriter keyboard", "umbrella",
    "unicycle", "upright", "vacuum", "vase", "vault",
    "velvet", "vending machine", "vestment", "viaduct", "violin",
    "volleyball", "waffle iron", "wall clock", "wallet", "wardrobe",
    "warplane", "washbasin", "washer", "water bottle", "water jug",
    "water tower", "whiskey jug", "whistle", "wig", "window screen",
    "window shade", "Windsor tie", "wine bottle", "wing", "wok",
    "wooden spoon", "wool", "worm fence", "wreck", "yawl",
    "yurt", "web site", "comic book", "crossword puzzle", "street sign",
    "traffic light", "book jacket", "menu", "plate", "guacamole",
    "consomme", "hot pot", "trifle", "ice cream", "ice lolly",
    "French loaf", "bagel", "pretzel", "cheeseburger", "hotdog",
    "mashed potato", "head cabbage", "broccoli", "cauliflower", "zucchini",
    "spaghetti squash", "acorn squash", "butternut squash", "cucumber", "artichoke",
    "bell pepper", "cardoon", "mushroom", "Granny Smith", "strawberry",
    "orange", "lemon", "fig", "pineapple", "banana",
    "jackfruit", "custard apple", "pomegranate", "hay", "carbonara",
    "chocolate sauce", "dough", "meat loaf", "pizza", "potpie",
    "burrito", "red wine", "espresso", "cup", "eggnog",
    "alp", "bubble", "cliff", "coral reef", "geyser",
    "lakeside", "promontory", "sandbar", "seashore", "valley",
    "volcano", "ballplayer", "groom", "scuba diver", "rapeseed",
    "daisy", "yellow lady's slipper", "corn", "acorn", "hip",
    "buckeye", "coral fungus", "agaric", "gyromitra", "stinkhorn",
    "earthstar", "hen-of-the-woods", "bolete", "ear", "toilet tissue",
}

// ============================================================================
// Classification Result Structure
// ============================================================================

struct ClassificationResult {
    int class_id;
    float probability;
    std::string class_name;

    ClassificationResult(int id, float prob, const std::string& name)
        : class_id(id), probability(prob), class_name(name) {}
};

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Convert BGR image to NV12 format
 */
cv::Mat bgr2nv12(const cv::Mat& bgr_img) {
    auto start = std::chrono::high_resolution_clock::now();

    int height = bgr_img.rows;
    int width = bgr_img.cols;

    // BGR to YUV420P
    cv::Mat yuv_mat;
    cv::cvtColor(bgr_img, yuv_mat, cv::COLOR_BGR2YUV_I420);
    uint8_t* yuv = yuv_mat.ptr<uint8_t>();

    // Allocate NV12 image
    cv::Mat nv12_img(height * 3 / 2, width, CV_8UC1);
    uint8_t* nv12 = nv12_img.ptr<uint8_t>();

    // Copy Y plane
    int y_size = height * width;
    memcpy(nv12, yuv, y_size);

    // Convert UV planar to UV packed (NV12)
    int uv_height = height / 2;
    int uv_width = width / 2;
    uint8_t* nv12_uv = nv12 + y_size;
    uint8_t* u_data = yuv + y_size;
    uint8_t* v_data = u_data + uv_height * uv_width;

    for (int i = 0; i < uv_width * uv_height; i++) {
        *nv12_uv++ = *u_data++;
        *nv12_uv++ = *v_data++;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    LOG_TIME("BGR to NV12 time", duration);

    return nv12_img;
}

/**
 * @brief Preprocess image with letterbox or resize
 */
cv::Mat preprocess_image(const cv::Mat& img, int input_h, int input_w,
                         float& x_scale, float& y_scale,
                         int& x_shift, int& y_shift) {
    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat result;

    if (PREPROCESS_TYPE == LETTERBOX_TYPE) {
        // Letterbox preprocessing
        x_scale = std::min(1.0f * input_h / img.rows, 1.0f * input_w / img.cols);
        y_scale = x_scale;

        if (x_scale <= 0 || y_scale <= 0) {
            throw std::runtime_error("Invalid scale factor");
        }

        int new_w = static_cast<int>(img.cols * x_scale);
        int new_h = static_cast<int>(img.rows * y_scale);

        x_shift = (input_w - new_w) / 2;
        y_shift = (input_h - new_h) / 2;
        int x_other = input_w - new_w - x_shift;
        int y_other = input_h - new_h - y_shift;

        cv::resize(img, result, cv::Size(new_w, new_h));
        cv::copyMakeBorder(result, result, y_shift, y_other, x_shift, x_other,
                          cv::BORDER_CONSTANT, cv::Scalar(127, 127, 127));

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
        LOG_TIME("Preprocess (LetterBox) time", duration);

    } else if (PREPROCESS_TYPE == RESIZE_TYPE) {
        // Resize preprocessing
        cv::resize(img, result, cv::Size(input_w, input_h));

        x_scale = 1.0f * input_w / img.cols;
        y_scale = 1.0f * input_h / img.rows;
        x_shift = 0;
        y_shift = 0;

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
        LOG_TIME("Preprocess (Resize) time", duration);
    }

    LOG_INFO("Scale: x=" << x_scale << ", y=" << y_scale);
    LOG_INFO("Shift: x=" << x_shift << ", y=" << y_shift);

    return result;
}

/**
 * @brief Softmax function
 */
std::vector<float> softmax(const std::vector<float>& logits) {
    std::vector<float> result(logits.size());

    // Find max for numerical stability
    float max_val = *std::max_element(logits.begin(), logits.end());

    // Compute exp and sum
    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); i++) {
        result[i] = std::exp(logits[i] - max_val);
        sum += result[i];
    }

    // Normalize
    for (size_t i = 0; i < logits.size(); i++) {
        result[i] /= sum;
    }

    return result;
}

/**
 * @brief Get top K classification results
 */
std::vector<ClassificationResult> get_topk_results(
    const std::vector<float>& probabilities, int k) {

    // Create index array
    std::vector<int> indices(probabilities.size());
    std::iota(indices.begin(), indices.end(), 0);

    // Partial sort to get top k
    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
        [&probabilities](int a, int b) {
            return probabilities[a] > probabilities[b];
        });

    // Create results
    std::vector<ClassificationResult> results;
    for (int i = 0; i < k; i++) {
        int idx = indices[i];
        std::string class_name = (idx < IMAGENET_CLASSES.size())
            ? IMAGENET_CLASSES[idx]
            : "class_" + std::to_string(idx);
        results.emplace_back(idx, probabilities[idx], class_name);
    }

    return results;
}

// ============================================================================
// Main Function
// ============================================================================

int main(int argc, char** argv) {
    LOG_INFO("=== Ultralytics YOLO Classify Demo (C++) ===");
    LOG_INFO("OpenCV Version: " << CV_VERSION);

    // ========================================================================
    // 0. Parse command line arguments
    // ========================================================================

    std::string model_path = MODEL_PATH;
    std::string test_img_path = TEST_IMG_PATH;

    if (argc >= 2) model_path = argv[1];
    if (argc >= 3) test_img_path = argv[2];

    // ========================================================================
    // 1. Load BPU model
    // ========================================================================

    LOG_INFO("Loading model: " << model_path);
    auto start_time = std::chrono::high_resolution_clock::now();

    hbPackedDNNHandle_t packed_dnn_handle;
    const char* model_file_name = model_path.c_str();
    CHECK_SUCCESS(
        hbDNNInitializeFromFiles(&packed_dnn_handle, &model_file_name, 1),
        "Failed to initialize model from file");

    auto load_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - start_time).count() / 1000.0;
    LOG_TIME("Load model time", load_duration);

    // ========================================================================
    // 2. Get model handle
    // ========================================================================

    const char** model_name_list;
    int model_count = 0;
    CHECK_SUCCESS(
        hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle),
        "Failed to get model name list");

    const char* model_name = model_name_list[0];
    LOG_INFO("Model name: " << model_name);

    hbDNNHandle_t dnn_handle;
    CHECK_SUCCESS(
        hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name),
        "Failed to get model handle");

    // ========================================================================
    // 3. Check model input
    // ========================================================================

    int32_t input_count = 0;
    CHECK_SUCCESS(
        hbDNNGetInputCount(&input_count, dnn_handle),
        "Failed to get input count");

    if (input_count != 1) {
        LOG_ERROR("Model should have exactly 1 input, but has " << input_count);
        return -1;
    }

    hbDNNTensorProperties input_properties;
    CHECK_SUCCESS(
        hbDNNGetInputTensorProperties(&input_properties, dnn_handle, 0),
        "Failed to get input tensor properties");

    // Check tensor type
    if (input_properties.tensorType != HB_DNN_IMG_TYPE_NV12) {
        LOG_ERROR("Input tensor type is not HB_DNN_IMG_TYPE_NV12");
        return -1;
    }
    LOG_INFO("Input tensor type: HB_DNN_IMG_TYPE_NV12");

    // Check tensor layout
    if (input_properties.tensorLayout != HB_DNN_LAYOUT_NCHW) {
        LOG_ERROR("Input tensor layout is not HB_DNN_LAYOUT_NCHW");
        return -1;
    }
    LOG_INFO("Input tensor layout: HB_DNN_LAYOUT_NCHW");

    // Get input shape
    if (input_properties.validShape.numDimensions != 4) {
        LOG_ERROR("Input tensor should have 4 dimensions");
        return -1;
    }

    int32_t input_h = input_properties.validShape.dimensionSize[2];
    int32_t input_w = input_properties.validShape.dimensionSize[3];
    LOG_INFO("Input shape: (1, 3, " << input_h << ", " << input_w << ")");

    // ========================================================================
    // 4. Check model outputs
    // ========================================================================

    int32_t output_count = 0;
    CHECK_SUCCESS(
        hbDNNGetOutputCount(&output_count, dnn_handle),
        "Failed to get output count");

    if (output_count != 1) {
        LOG_ERROR("Classification model should have exactly 1 output, but has " << output_count);
        return -1;
    }

    hbDNNTensorProperties output_properties;
    CHECK_SUCCESS(
        hbDNNGetOutputTensorProperties(&output_properties, dnn_handle, 0),
        "Failed to get output tensor properties");

    LOG_INFO("Output shape: ("
             << output_properties.validShape.dimensionSize[0] << ", "
             << output_properties.validShape.dimensionSize[1] << ", "
             << output_properties.validShape.dimensionSize[2] << ", "
             << output_properties.validShape.dimensionSize[3] << ")");

    int num_classes = output_properties.validShape.dimensionSize[1];
    LOG_INFO("Number of classes: " << num_classes);

    // ========================================================================
    // 5. Load and preprocess image
    // ========================================================================

    LOG_INFO("Loading image: " << test_img_path);
    cv::Mat img = cv::imread(test_img_path);
    if (img.empty()) {
        LOG_ERROR("Failed to load image: " << test_img_path);
        return -1;
    }
    LOG_INFO("Image size: " << img.cols << "x" << img.rows);

    // Preprocess image
    float x_scale, y_scale;
    int x_shift, y_shift;
    cv::Mat preprocessed = preprocess_image(img, input_h, input_w,
                                           x_scale, y_scale, x_shift, y_shift);

    // Convert to NV12
    cv::Mat nv12_img = bgr2nv12(preprocessed);

    // ========================================================================
    // 6. Prepare input tensor
    // ========================================================================

    hbDNNTensor input;
    input.properties = input_properties;

    int input_memSize = input_h * input_w * 3 / 2;
    hbSysAllocCachedMem(&input.sysMem[0], input_memSize);
    memcpy(input.sysMem[0].virAddr, nv12_img.ptr<uint8_t>(), input_memSize);
    hbSysFlushMem(&input.sysMem[0], HB_SYS_MEM_CACHE_CLEAN);

    // ========================================================================
    // 7. Prepare output tensor
    // ========================================================================

    hbDNNTensor* output = new hbDNNTensor[1];
    output[0].properties = output_properties;
    int out_size = output_properties.alignedByteSize;
    hbSysAllocCachedMem(&output[0].sysMem[0], out_size);

    // ========================================================================
    // 8. Run inference
    // ========================================================================

    LOG_INFO("Running inference...");
    start_time = std::chrono::high_resolution_clock::now();

    hbDNNTaskHandle_t task_handle = nullptr;
    hbDNNInferCtrlParam infer_ctrl_param;
    HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&infer_ctrl_param);

    hbDNNInfer(&task_handle, &output, &input, dnn_handle, &infer_ctrl_param);
    hbDNNWaitTaskDone(task_handle, 0);

    auto infer_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - start_time).count() / 1000.0;
    LOG_TIME("BPU inference time", infer_duration);

    // ========================================================================
    // 9. Post-process
    // ========================================================================

    LOG_INFO("Post-processing...");
    start_time = std::chrono::high_resolution_clock::now();

    // Flush memory
    hbSysFlushMem(&output[0].sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);

    // Get output data
    float* output_data = reinterpret_cast<float*>(output[0].sysMem[0].virAddr);

    // Convert to vector
    std::vector<float> logits(output_data, output_data + num_classes);

    // Apply softmax
    std::vector<float> probabilities = softmax(logits);

    // Get top K results
    std::vector<ClassificationResult> results = get_topk_results(probabilities, TOP_K);

    auto post_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - start_time).count() / 1000.0;
    LOG_TIME("Post-processing time", post_duration);

    // ========================================================================
    // 10. Display results
    // ========================================================================

    LOG_INFO("Classification results:");
    LOG_INFO("Image: " << test_img_path);
    std::cout << std::endl;

    for (size_t i = 0; i < results.size(); i++) {
        const auto& res = results[i];
        std::cout << "\033[1;32m"
                  << "TOP" << (i + 1) << " -> "
                  << "id: " << res.class_id << ", "
                  << "score: " << std::fixed << std::setprecision(3) << res.probability << ", "
                  << "name: " << res.class_name
                  << "\033[0m" << std::endl;
    }

    // ========================================================================
    // 11. Cleanup
    // ========================================================================

    hbDNNReleaseTask(task_handle);
    hbSysFreeMem(&input.sysMem[0]);
    hbSysFreeMem(&output[0].sysMem[0]);
    delete[] output;
    hbDNNRelease(packed_dnn_handle);

    LOG_INFO("=== Demo completed successfully ===");
    return 0;
}
