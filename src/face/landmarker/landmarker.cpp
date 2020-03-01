#include "landmarker.h"
#include "pfldlandmarker/pfldlandmarker.h"
#include "zqlandmarker/zqlandmarker.h"

namespace mirror {
Landmarker* PFLDLandmarkerFactory::CreateLandmarker() {
    return new PFLDLandmarker();
}

Landmarker* ZQLandmarkerFactory::CreateLandmarker() {
    return new ZQLandmarker();
}

}