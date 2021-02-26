//Author: Daniil Pavlov

#include "CudaDomainDecomposition.h"
#include "CudaKernelSources.h"

using namespace OpenMM;
using namespace std;

CudaDDUtilities::CudaDDUtilities(CudaPlatform::PlatformData& data, const System& system, ContextImpl& contextImpl) : data(data), system(system) {
    molecules = contextImpl.getMolecules();
    moleculeInd.resize(system.getNumParticles());

    for(int i = 0; i < molecules.size(); i++)
        for(int j : molecules[i])
            moleculeInd[j] = i;
}


CudaDDCalcForcesAndEnergyKernel::CudaDDCalcForcesAndEnergyKernel(string name, const Platform& platform, CudaPlatform::PlatformData& data) : CalcForcesAndEnergyKernel(name, platform), data(data) {
}

void CudaDDCalcForcesAndEnergyKernel::initialize(const System& system) {
    //TODO
}

void CudaDDCalcForcesAndEnergyKernel::beginComputation(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
    //TODO
}

double CudaDDCalcForcesAndEnergyKernel::finishComputation(ContextImpl& context, bool includeForces, bool includeEnergy, int groups, bool& valid) {
    //TODO
}


CudaDDUpdateStateDataKernel::CudaDDUpdateStateDataKernel(string name, const Platform& platform, CudaPlatform::PlatformData& data) : UpdateStateDataKernel(name, platform), data(data) {
}

void CudaDDUpdateStateDataKernel::initialize(const System& system) {
    //TODO
}

double CudaDDUpdateStateDataKernel::getTime(const ContextImpl& context) const {
    return data.contexts[0]->getTime();
}

void CudaDDUpdateStateDataKernel::setTime(ContextImpl& context, double time) {
    for (auto ctx : data.contexts)
        ctx->setTime(time);
}

void CudaDDUpdateStateDataKernel::getPositions(ContextImpl& context, vector<Vec3>& positions) {
    //TODO
}

void CudaDDUpdateStateDataKernel::setPositions(ContextImpl& context, const vector<Vec3>& positions) {
    //TODO
}

void CudaDDUpdateStateDataKernel::getVelocities(ContextImpl& context, vector<Vec3>& velocities) {
    //TODO
}

void CudaDDUpdateStateDataKernel::setVelocities(ContextImpl& context, const vector<Vec3>& velocities) {
    //TODO
}

void CudaDDUpdateStateDataKernel::getForces(ContextImpl& context, vector<Vec3>& forces) {
    //TODO
}

void CudaDDUpdateStateDataKernel::getEnergyParameterDerivatives(ContextImpl& context, map<string, double>& derivs) {
    //TODO
}

void CudaDDUpdateStateDataKernel::getPeriodicBoxVectors(ContextImpl& context, Vec3& a, Vec3& b, Vec3& c) const {
    data.contexts[0]->getPeriodicBoxVectors(a, b, c);
}

void CudaDDUpdateStateDataKernel::setPeriodicBoxVectors(ContextImpl& context, const Vec3& a, const Vec3& b, const Vec3& c) {
    // If any particles have been wrapped to the first periodic box, we need to unwrap them
    // to avoid changing their positions.

    vector<Vec3> positions;
    for (auto& offset : data.contexts[0]->getPosCellOffsets()) {
        if (offset.x != 0 || offset.y != 0 || offset.z != 0) {
            getPositions(context, positions);
            break;
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


CudaDDApplyConstraintsKernel::CudaDDApplyConstraintsKernel(string name, const Platform& platform, CudaPlatform::PlatformData& data) : ApplyConstraintsKernel(name, platform), data(data) {
}

void CudaDDApplyConstraintsKernel::initialize(const System& system) {
    //TODO
}

void CudaDDApplyConstraintsKernel::apply(ContextImpl& context, double tol) {
    //TODO
}

void CudaDDApplyConstraintsKernel::applyToVelocities(ContextImpl& context, double tol) {
    //TODO
}


CudaDDVirtualSitesKernel::CudaDDVirtualSitesKernel(string name, const Platform& platform, CudaPlatform::PlatformData& data) : VirtualSitesKernel(name, platform), data(data) {
}

void CudaDDVirtualSitesKernel::initialize(const System& system) {
    //TODO
}

void CudaDDVirtualSitesKernel::computePositions(ContextImpl& context) {
    //TODO
}


CudaDDCalcNonbondedForceKernel::CudaDDCalcNonbondedForceKernel(string name, const Platform& platform, CudaPlatform::PlatformData& data, const System& system) : CalcNonbondedForceKernel(name, platform), data(data) {
}

void CudaDDCalcNonbondedForceKernel::initialize(const System& system, const NonbondedForce& force) {
    NonbondedMethod nonbondedMethod = CalcNonbondedForceKernel::NonbondedMethod(force.getNonbondedMethod());
    if(nonbondedMethod == NoCutoff)
        throw OpenMMException("Domain decomposition requires a cutoff.");
    if(nonbondedMethod == Ewald)
        throw OpenMMException("Ewald summation for domain decomposition is not implemented yet.");
    if(nonbondedMethod == PME || nonbondedMethod == LJPME)
        throw OpenMMException("PME for domain decomposition is not implemented yet.");
    //TODO
}

double CudaDDCalcNonbondedForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy, bool includeDirect, bool includeReciprocal) {
    //TODO
}

void CudaDDCalcNonbondedForceKernel::copyParametersToContext(ContextImpl& context, const NonbondedForce& force) {
    //TODO
}

void CudaDDCalcNonbondedForceKernel::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    throw OpenMMException("getPMEParametersInContext: PME for domain decomposition is not implemented yet.");
}

void CudaDDCalcNonbondedForceKernel::getLJPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    throw OpenMMException("getPMEParametersInContext: PME for domain decomposition is not implemented yet.");
}


CudaDDIntegrateVerletStepKernel::CudaDDIntegrateVerletStepKernel(string name, const Platform& platform, CudaPlatform::PlatformData& data) : IntegrateVerletStepKernel(name, platform), data(data) {
}

void CudaDDIntegrateVerletStepKernel::initialize(const System& system, const VerletIntegrator& integrator) {
    //TODO
}

void CudaDDIntegrateVerletStepKernel::execute(ContextImpl& context, const VerletIntegrator& integrator) {
    //TODO
}

double CudaDDIntegrateVerletStepKernel::computeKineticEnergy(ContextImpl& context, const VerletIntegrator& integrator) {
    //TODO
}