#include "recognizer.h"
#include "./mobilefacenet/mobilefacenet.h"

namespace mirror {
Recognizer* MobilefacenetFactory::CreateRecognizer() {
    return new Mobilefacenet();
}

}