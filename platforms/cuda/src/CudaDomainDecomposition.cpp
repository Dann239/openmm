//Author: Daniil Pavlov

#include "CudaDomainDecomposition.h"
#include "CudaKernelSources.h"

using namespace OpenMM;
using namespace std;

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
        const System& system = data.ddutilities->getSubsystems()[i];
        CudaContext& cu = *data.contexts[i];

        if (name == CalcForcesAndEnergyKernel::Name()) {
            auto newKernel = new CudaCalcForcesAndEnergyKernel(name, platform, cu);
            newKernel->initialize(system);
            kernel = newKernel;
        }
        // CudaDDUpdateStateData doesn't implement CudaDDInterface because reasons
        if (name == ApplyConstraintsKernel::Name()) {
            auto newKernel = new CudaApplyConstraintsKernel(name, platform, cu);
            newKernel->initialize(system);
            kernel = newKernel;
        }
        if (name == VirtualSitesKernel::Name()) {
            auto newKernel = new CudaVirtualSitesKernel(name, platform, cu);
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
    data(data), system(system), updater(nullptr), time(0) {

    system.getDefaultPeriodicBoxVectors(box[0], box[1], box[2]);

    molecules = contextImpl.getMolecules();
    moleculeInd.resize(system.getNumParticles());

    for (int i = 0; i < molecules.size(); i++)
        for (int j : molecules[i])
            moleculeInd[j] = i;
}

const vector<System>& CudaDDUtilities::getSubsystems() {
    if (positions.size() == 0)
        throw OpenMMException("Domain decomposition cannot be performed, particle positions have not been set!");
    if (subsystems.size() == 0) {
        subsystems.resize(data.devices.size());
        //TODO domain decomposition
    }
    return subsystems;
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
            const System& system = getSubsystems()[i];
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
}

void CudaDDUtilities::destroyContexts() {
    for (int i = 0; i < registeredKernels.size(); i++)
        registeredKernels[i]->destroyKernels();
    for (int i = 0; i < data.contexts.size(); i++)
        delete data.contexts[i];
}


void CudaDDCalcForcesAndEnergyKernel::beginComputation(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
    for (int i = 0; i < data.contexts.size(); i++)
        getKernel(i).beginComputation(context, includeForces, includeEnergy, groups);
}

double CudaDDCalcForcesAndEnergyKernel::finishComputation(ContextImpl& context, bool includeForces, bool includeEnergy, int groups, bool& valid) {
    for (int i = 0; i < data.contexts.size(); i++)
        getKernel(i).finishComputation(context, includeForces, includeEnergy, groups, valid);
    //TODO gather energy
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
    data.ddutilities->prepareContexts();
    //TODO spread over domains
}

void CudaDDUpdateStateDataKernel::getVelocities(ContextImpl& context, vector<Vec3>& velocities) {
    //TODO gather domains
}

void CudaDDUpdateStateDataKernel::setVelocities(ContextImpl& context, const vector<Vec3>& velocities) {
    data.ddutilities->velocities = velocities;
    if(data.contexts.size() == 0)
        return;
    //TODO spread over domains
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
    for (int i = 0; i < kernels.size(); i++)
        getKernel(i).apply(context, tol);
}

void CudaDDApplyConstraintsKernel::applyToVelocities(ContextImpl& context, double tol) {
    for (int i = 0; i < kernels.size(); i++)
        getKernel(i).applyToVelocities(context, tol);
}


void CudaDDVirtualSitesKernel::computePositions(ContextImpl& context) {
    for (int i = 0; i < kernels.size(); i++)
        getKernel(i).computePositions(context);
}


double CudaDDCalcNonbondedForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy, bool includeDirect, bool includeReciprocal) {
    for (int i = 0; i < kernels.size(); i++)
        getKernel(i).execute(context, includeForces, includeEnergy, includeDirect, includeReciprocal);
}

void CudaDDCalcNonbondedForceKernel::copyParametersToContext(ContextImpl& context, const NonbondedForce& force) {
    //TODO spread over domains
    for (int i = 0; i < kernels.size(); i++)
        getKernel(i).copyParametersToContext(context, force);
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
    //TODO spread over domains
    for (int i = 0; i < kernels.size(); i++)
        getKernel(i).initialize(data.ddutilities->getSubsystems()[i], *storedForce);
}


void CudaDDIntegrateVerletStepKernel::execute(ContextImpl& context, const VerletIntegrator& integrator) {
    for (int i = 0; i < kernels.size(); i++)
        getKernel(i).execute(context, integrator);
    //TODO halo exchange
}

double CudaDDIntegrateVerletStepKernel::computeKineticEnergy(ContextImpl& context, const VerletIntegrator& integrator) {
    //TODO gather energy
}

void CudaDDIntegrateVerletStepKernel::prepareKernels() {
    CudaDDInterface::prepareKernels();
    if (!storedIntegrator)
        throw OpenMMException("Integrator hasn't been initialized!");
    for (int i = 0; i < kernels.size(); i++)
        getKernel(i).initialize(data.ddutilities->getSubsystems()[i], *storedIntegrator);
}
