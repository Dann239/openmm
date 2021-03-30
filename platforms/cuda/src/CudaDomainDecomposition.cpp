//Author: Daniil Pavlov

#include "CudaDomainDecomposition.h"

#include <algorithm>

using namespace OpenMM;
using namespace std;


template<typename TBase, typename TMethod, typename ...Args>
void forEachKernel(CudaPlatform::PlatformData& data, TBase* caller, TMethod method, Args ...args) {
    for(int i = 0; i < data.contexts.size(); i++) {
        data.contexts[i]->setAsCurrent();
        (caller->getKernel(i).*method)(args...);
    }
}

CudaDDInterface::CudaDDInterface(std::string name, const Platform& platform, CudaPlatform::PlatformData& data) : name(name), platform(platform), data(data) {
    data.ddutilities->registerKernel(this);
}

void CudaDDInterface::destroyKernels() {
    kernels.clear();
}

void CudaDDInterface::prepareKernels() {
    if(data.contexts.size() == 0)
        data.ddutilities->prepareContexts();
    if (kernels.size() == data.contexts.size())
        return;
    if (kernels.size() != 0)
        throw OpenMMException("Kernel array in CudaDDInterface has an invalid size!");
    for (int i = 0; i < data.contexts.size(); i++) {
        KernelImpl* kernel = nullptr;
        const System& system = data.ddutilities->system;
        CudaContext& cu = *data.contexts[i];

        if (name == CalcForcesAndEnergyKernel::Name()) {
            auto newKernel = new CudaCalcForcesAndEnergyKernel(name, platform, cu);
            newKernel->initialize(system);
            kernel = newKernel;
        }
        if (name == UpdateStateDataKernel::Name()) {
            auto newKernel = new CudaUpdateStateDataKernel(name, platform, cu);
            newKernel->initialize(system);
            kernel = newKernel;
        }
        if (name == ApplyConstraintsKernel::Name()) {
            auto newKernel = new CommonApplyConstraintsKernel(name, platform, cu);
            newKernel->initialize(system);
            kernel = newKernel;
        }
        if (name == VirtualSitesKernel::Name()) {
            auto newKernel = new CommonVirtualSitesKernel(name, platform, cu);
            newKernel->initialize(system);
            kernel = newKernel;
        }
        if (name == CalcNonbondedForceKernel::Name()) {
            kernel = new CudaCalcNonbondedForceKernel(name, platform, cu, system);
        }
        if (name == IntegrateVerletStepKernel::Name()) {
            kernel = new CommonIntegrateVerletStepKernel(name, platform, cu);
        }

        kernels.emplace_back(kernel);
    }
}


CudaDDUtilities::CudaDDUtilities(CudaPlatform::PlatformData& data, const System& system, ContextImpl& contextImpl) :
    data(data), system(system), contextImpl(contextImpl), updater(nullptr), time(0), cutoff(0) {

    numAtoms = system.getNumParticles();
    paddedNumAtoms = (numAtoms + 31)/32*32;

    domainMasks.resize(data.devices.size(), vector<unsigned int>(paddedNumAtoms / 32));
    enabledMasks.resize(data.devices.size(), vector<unsigned int>(paddedNumAtoms / 32));
    domainInd.resize(numAtoms, -1);

    system.getDefaultPeriodicBoxVectors(box[0], box[1], box[2]);

    molecules = contextImpl.getMolecules();
    moleculeInd.resize(numAtoms);

    for (int i = 0; i < molecules.size(); i++)
        for (int j : molecules[i])
            moleculeInd[j] = i;
}

void CudaDDUtilities::registerKernel(CudaDDInterface* kernel) {
    registeredKernels.push_back(kernel);
}

void CudaDDUtilities::registerUpdater(CudaDDUpdateStateDataKernel* kernel) {
    if (updater)
        throw OpenMMException("CudaDDUtilities: updater already registered.");
    updater = kernel;
}

void CudaDDUtilities::prepareContexts() {
    if (data.devices.size() == data.contexts.size())
        return;
    if (data.contexts.size() != 0)
        throw OpenMMException("CudaContext array in CudaPlatform::PlatformData has an invalid size!");
    for (int i = 0; i < data.devices.size(); i++) {
        if (data.devices[i].length() > 0) {
            int deviceIndex = stoi(data.devices[i]);
            bool blocking = (data.propertyValues[CudaPlatform::CudaUseBlockingSync()] == "true");
            const string& precisionProperty = data.propertyValues[CudaPlatform::CudaPrecision()];
            const string& compilerProperty = data.propertyValues[CudaPlatform::CudaCompiler()];
            const string& tempProperty = data.propertyValues[CudaPlatform::CudaTempDirectory()];
            const string& hostCompilerProperty = data.propertyValues[CudaPlatform::CudaHostCompiler()];
            bool allowRuntimeCompiler = data.allowRuntimeCompiler;
            data.contexts.push_back(new CudaContext(system, deviceIndex, blocking, precisionProperty, compilerProperty, tempProperty, hostCompilerProperty, allowRuntimeCompiler, data, nullptr));
        }
    }
    for(int i = 0; i < registeredKernels.size(); i++)
        registeredKernels[i]->prepareKernels();
    decompose();
    updater->setPositions(contextImpl, positions);
    updater->setVelocities(contextImpl, velocities);
    updater->setPeriodicBoxVectors(contextImpl, box[0], box[1], box[2]);
    updater->setTime(contextImpl, time);
}

void CudaDDUtilities::destroyContexts() {
    for (int i = 0; i < registeredKernels.size(); i++)
        registeredKernels[i]->destroyKernels();
    for (int i = 0; i < data.contexts.size(); i++)
        delete data.contexts[i];
}

void CudaDDUtilities::decompose() {
    if (positions.size() == 0)
        throw OpenMMException("Domain decomposition cannot be performed, particle positions have not been set!");
    // For now, we just split along the z-axis and do no load-balancing whatsoever
    // TODO enhance the algorithm
    resetCutoff();
    for(int i = 0; i < data.devices.size(); i++) {
        domains.emplace_back(Domain{
            0., box[0][0] + box[1][0] + box[2][0],
            0., box[1][1] + box[2][1],
            box[2][2] / data.devices.size() * i, box[2][2] / data.devices.size() * (i + 1)
        });
    }
    fill(domainInd.begin(), domainInd.end(), -1);
    for(int i = 0; i < data.devices.size(); i++) {
        fill(domainMasks[i].begin(), domainMasks[i].end(), 0);
        fill(enabledMasks[i].begin(), enabledMasks[i].end(), 0);
    }
    for(int i = 0; i < numAtoms; i++) {
        Vec3 pos = positions[i];
        double scale3 = floor(pos[2] / box[2][2]);
        pos[0] -= scale3*box[2][0];
        pos[1] -= scale3*box[2][1];
        pos[2] -= scale3*box[2][2];
        double scale2 = floor(pos[1] / box[1][1]);
        pos[0] -= scale2*box[1][0];
        pos[1] -= scale2*box[1][1];
        double scale1 = floor(pos[0] / box[0][0]);
        pos[0] -= scale1*box[0][0];

        auto& molecule = molecules[moleculeInd[i]];
        for(int j = 0; j < domains.size(); j++) {
            if( pos[0] > domains[j].xlo && pos[0] <= domains[j].xhi &&
                pos[1] > domains[j].ylo && pos[1] <= domains[j].yhi &&
                pos[2] > domains[j].zlo && pos[2] <= domains[j].zhi
              ) {
                domainMasks[j][i / 32] |= 1 << (i % 32);
                domainInd[i] = j;
            }
            if( pos[0] >= domains[j].xlo - cutoff && pos[0] <= domains[j].xhi + cutoff &&
                pos[1] >= domains[j].ylo - cutoff && pos[1] <= domains[j].yhi + cutoff &&
                pos[2] >= domains[j].zlo - cutoff && pos[2] <= domains[j].zhi + cutoff
              ) {
                for(int k = 0; k < molecule.size(); k++)
                    enabledMasks[j][molecule[k] / 32] |= 1 << (molecule[k] % 32);
            }
        }
        if(domainInd[i] == -1)
            throw OpenMMException("Particle " + to_string(i) + " was not assigned to a domain.");
    }
    // TODO save into CudaNonbondedUtilities
}

void CudaDDUtilities::resetCutoff() {
    cutoff = data.contexts[0]->getNonbondedUtilities().getMaxCutoffDistance();
}

double CudaDDUtilities::getCutoff() {
    return cutoff;
}


void CudaDDCalcForcesAndEnergyKernel::beginComputation(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
    forEachKernel(data, this, &CalcForcesAndEnergyKernel::beginComputation, context, includeForces, includeEnergy, groups);
}

double CudaDDCalcForcesAndEnergyKernel::finishComputation(ContextImpl& context, bool includeForces, bool includeEnergy, int groups, bool& valid) {
    forEachKernel(data, this, &CalcForcesAndEnergyKernel::finishComputation, context, includeForces, includeEnergy, groups, valid);
    //TODO gather energy
}


CudaDDUpdateStateDataKernel::CudaDDUpdateStateDataKernel(std::string name, const Platform& platform, CudaPlatform::PlatformData& data) :
    UpdateStateDataKernel(name, platform), data(data), CudaDDInterface(name, platform, data) {
    data.ddutilities->registerUpdater(this);
}

double CudaDDUpdateStateDataKernel::getTime(const ContextImpl& context) const {
    return data.ddutilities->time;
}

void CudaDDUpdateStateDataKernel::setTime(ContextImpl& context, double time) {
    data.ddutilities->time = time;
    for (auto ctx : data.contexts)
        ctx->setTime(time);
}

void CudaDDUpdateStateDataKernel::getPositions(ContextImpl& context, vector<Vec3>& positions) {
    //TODO gather domains
}

void CudaDDUpdateStateDataKernel::setPositions(ContextImpl& context, const vector<Vec3>& positions) {
    data.ddutilities->positions = positions;
    if(data.contexts.size() == 0) {
        data.ddutilities->prepareContexts();
        return;
    }
    else {
        data.ddutilities->decompose();
    }
    forEachKernel(data, this, &UpdateStateDataKernel::setPositions, context, positions);
}

void CudaDDUpdateStateDataKernel::getVelocities(ContextImpl& context, vector<Vec3>& velocities) {
    //TODO gather domains
}

void CudaDDUpdateStateDataKernel::setVelocities(ContextImpl& context, const vector<Vec3>& velocities) {
    data.ddutilities->velocities = velocities;
    forEachKernel(data, this, &UpdateStateDataKernel::setVelocities, context, velocities);
}

void CudaDDUpdateStateDataKernel::getForces(ContextImpl& context, vector<Vec3>& forces) {
    //TODO gather domains
}

void CudaDDUpdateStateDataKernel::getEnergyParameterDerivatives(ContextImpl& context, map<string, double>& derivs) {
    //TODO gather energy derivs
}

void CudaDDUpdateStateDataKernel::getPeriodicBoxVectors(ContextImpl& context, Vec3& a, Vec3& b, Vec3& c) const {
    a = data.ddutilities->box[0];
    b = data.ddutilities->box[1];
    c = data.ddutilities->box[2];
}

void CudaDDUpdateStateDataKernel::setPeriodicBoxVectors(ContextImpl& context, const Vec3& a, const Vec3& b, const Vec3& c) {
    data.ddutilities->box[0] = a;
    data.ddutilities->box[1] = b;
    data.ddutilities->box[2] = c;
    if(data.contexts.size() == 0)
        return;

    // If any particles have been wrapped to the first periodic box, we need to unwrap them
    // to avoid changing their positions.

    vector<Vec3> positions;
    for (auto ctx : data.contexts) {
        for (auto& offset : ctx->getPosCellOffsets()) {
            if (offset.x != 0 || offset.y != 0 || offset.z != 0) {
                getPositions(context, positions);
                break;
            }
        }
    }

    // Update the vectors.
    for (auto ctx : data.contexts)
        ctx->setPeriodicBoxVectors(a, b, c);
    if (positions.size() > 0)
        setPositions(context, positions);
}

void CudaDDUpdateStateDataKernel::createCheckpoint(ContextImpl& context, ostream& stream) {
    throw OpenMMException("Checkpoints for domain decomposition are not implemented yet.");
}

void CudaDDUpdateStateDataKernel::loadCheckpoint(ContextImpl& context, istream& stream) {
    throw OpenMMException("Checkpoints for domain decomposition are not implemented yet.");
}


void CudaDDApplyConstraintsKernel::apply(ContextImpl& context, double tol) {
    forEachKernel(data, this, &ApplyConstraintsKernel::apply, context, tol);
}

void CudaDDApplyConstraintsKernel::applyToVelocities(ContextImpl& context, double tol) {
    forEachKernel(data, this, &ApplyConstraintsKernel::applyToVelocities, context, tol);
}


void CudaDDVirtualSitesKernel::computePositions(ContextImpl& context) {
    forEachKernel(data, this, &VirtualSitesKernel::computePositions, context);
}


double CudaDDCalcNonbondedForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy, bool includeDirect, bool includeReciprocal) {
    forEachKernel(data, this, &CalcNonbondedForceKernel::execute, context, includeForces, includeEnergy, includeDirect, includeReciprocal);
}

void CudaDDCalcNonbondedForceKernel::copyParametersToContext(ContextImpl& context, const NonbondedForce& force) {
    double oldCutoff = data.ddutilities->getCutoff();
    forEachKernel(data, this, &CalcNonbondedForceKernel::copyParametersToContext, context, force);
    data.ddutilities->resetCutoff();
    if(oldCutoff != data.ddutilities->getCutoff())
        data.ddutilities->decompose();
}

void CudaDDCalcNonbondedForceKernel::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    throw OpenMMException("getPMEParametersInContext: PME for domain decomposition is not implemented yet.");
}

void CudaDDCalcNonbondedForceKernel::getLJPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    throw OpenMMException("getPMEParametersInContext: PME for domain decomposition is not implemented yet.");
}

void CudaDDCalcNonbondedForceKernel::prepareKernels() {
    CudaDDInterface::prepareKernels();
    if (!storedForce)
        throw OpenMMException("Force hasn't been initialized!");
    NonbondedMethod nonbondedMethod = CalcNonbondedForceKernel::NonbondedMethod(storedForce->getNonbondedMethod());
    if (nonbondedMethod == NoCutoff)
        throw OpenMMException("Domain decomposition requires a cutoff.");
    if (nonbondedMethod == Ewald)
        throw OpenMMException("Ewald summation for domain decomposition is not implemented yet.");
    if (nonbondedMethod == PME || nonbondedMethod == LJPME)
        throw OpenMMException("PME for domain decomposition is not implemented yet.");
    forEachKernel(data, this, &CalcNonbondedForceKernel::initialize, data.ddutilities->system, *storedForce);
}


void CudaDDIntegrateVerletStepKernel::execute(ContextImpl& context, const VerletIntegrator& integrator) {
    forEachKernel(data, this, &IntegrateVerletStepKernel::execute, context, integrator);
    //TODO halo exchange
}

double CudaDDIntegrateVerletStepKernel::computeKineticEnergy(ContextImpl& context, const VerletIntegrator& integrator) {
    //TODO gather energy
}

void CudaDDIntegrateVerletStepKernel::prepareKernels() {
    CudaDDInterface::prepareKernels();
    if (!storedIntegrator)
        throw OpenMMException("Integrator hasn't been initialized!");
    forEachKernel(data, this, &IntegrateVerletStepKernel::initialize, data.ddutilities->system, *storedIntegrator);
}
