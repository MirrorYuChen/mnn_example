#include "detecter.h"
#include "./centerface/centerface.h"

namespace mirror {
Detecter* CenterfaceFactory::CreateDetecter() {
    return new Centerface();
}


}