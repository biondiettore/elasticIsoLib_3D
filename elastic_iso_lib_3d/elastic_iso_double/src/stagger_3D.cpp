#include <stagger_3D.h>
using namespace SEP;

/******************************************************************************/
staggerZ::staggerZ(const std::shared_ptr<double3DReg> model, const std::shared_ptr<double3DReg> data){
  if( not data->getHyper()->getAxis(1).n == model->getHyper()->getAxis(1).n){
		throw std::runtime_error("**** ERROR [staggerZ-init]: inconsistent nz ****");
	}
  if( not data->getHyper()->getAxis(2).n == model->getHyper()->getAxis(2).n){
		throw std::runtime_error("**** ERROR [staggerZ-int]: inconsistent nx ****");
	}
	if( not data->getHyper()->getAxis(3).n == model->getHyper()->getAxis(3).n){
		throw std::runtime_error("**** ERROR [staggerZ-init]: inconsistent ny ****");
	}

	_ny = model->getHyper()->getAxis(3).n;
  _nx = model->getHyper()->getAxis(2).n;
  _nz = model->getHyper()->getAxis(1).n;

  setDomainRange(model,data);
}

void staggerZ::forward(const bool add,const  std::shared_ptr<double3DReg> model,  std::shared_ptr<double3DReg> data) const {
  if( not checkDomainRange(model,data)){
		throw std::runtime_error("**** ERROR [staggerZ-forward]: inconsistent vectors with operator's domain and range ****");
	};
  if(!add) data->scale(0.);

  #pragma omp parallel for collapse(3)
	for(int iy = 0; iy < _ny; iy++){
	  for(int ix = 0; ix < _nx; ix++){
	    for(int iz = 0; iz < _nz-1; iz++){
	      (*data->_mat)[iy][ix][iz] += 0.5 * ((*model->_mat)[iy][ix][iz] + (*model->_mat)[iy][ix][iz+1] );
	    }
	  }
	}

  //handle grid boundary on bottom side
  #pragma omp parallel for collapse(2)
	for(int iy = 0; iy < _ny; iy++){
	  for(int ix = 0; ix < _nx; ix++){
	    (*data->_mat)[iy][ix][_nz-1] += (*model->_mat)[iy][ix][_nz-1];
	  }
	}
}

void staggerZ::adjoint(const bool add, std::shared_ptr<double3DReg> model, const std::shared_ptr<double3DReg> data) const{
  if( not checkDomainRange(model,data)){
		throw std::runtime_error("**** ERROR [staggerZ-adjoint]: inconsistent vectors with operator's domain and range ****");
	};
  if(!add) model->scale(0.);

  #pragma omp parallel for collapse(3)
	for(int iy = 0; iy < _ny; iy++){
	  for(int ix = 0; ix < _nx; ix++){
	    for(int iz = 1; iz < _nz-1; iz++){
	      (*model->_mat)[iy][ix][iz] += 0.5 * ((*data->_mat)[iy][ix][iz] + (*data->_mat)[iy][ix][iz-1] );
	    }
	  }
	}

  //handle grid boundary on top and bottom
  #pragma omp parallel for collapse(2)
	for(int iy = 0; iy < _ny; iy++){
	  for(int ix = 0; ix < _nx; ix++){
	    (*model->_mat)[iy][ix][0] += 0.5 * (*data->_mat)[iy][ix][0]; //top
	    (*model->_mat)[iy][ix][_nz-1] += (*data->_mat)[iy][ix][_nz-1] + 0.5 * (*data->_mat)[iy][ix][_nz-2]; //bottom
	  }
	}
}
/******************************************************************************/

/******************************************************************************/
staggerX::staggerX(const std::shared_ptr<double3DReg> model, const std::shared_ptr<double3DReg> data){
  if( not data->getHyper()->getAxis(1).n == model->getHyper()->getAxis(1).n){
		throw std::runtime_error("**** ERROR [staggerX-init]: inconsistent nz ****");
	}
  if( not data->getHyper()->getAxis(2).n == model->getHyper()->getAxis(2).n){
		throw std::runtime_error("**** ERROR [staggerX-init]: inconsistent nx ****");
	}
	if( not data->getHyper()->getAxis(3).n == model->getHyper()->getAxis(3).n){
		throw std::runtime_error("**** ERROR [staggerX-init]: inconsistent ny ****");
	}

	_ny = model->getHyper()->getAxis(3).n;
  _nx = model->getHyper()->getAxis(2).n;
  _nz = model->getHyper()->getAxis(1).n;

  setDomainRange(model,data);
}

void staggerX::forward(const bool add,const  std::shared_ptr<double3DReg> model,  std::shared_ptr<double3DReg> data) const{
  if( not checkDomainRange(model,data)){
		throw std::runtime_error("**** ERROR [staggerX-forward]: inconsistent vectors with operator's domain and range ****");
	};
  if(!add) data->scale(0.);

  #pragma omp parallel for collapse(3)
	for(int iy = 0; iy < _ny; iy++){
	  for(int ix = 0; ix < _nx-1; ix++){
	    for(int iz = 0; iz < _nz; iz++){
	      (*data->_mat)[iy][ix][iz] += 0.5 * ((*model->_mat)[iy][ix][iz] + (*model->_mat)[iy][ix+1][iz] );
	    }
	  }
	}

  //handle grid boundary on right side
  #pragma omp parallel for collapse(2)
	for(int iy = 0; iy < _ny; iy++){
	  for(int iz = 0; iz < _nz; iz++){
	    (*data->_mat)[iy][_nx-1][iz] += (*model->_mat)[iy][_nx-1][iz];
	  }
	}
}

void staggerX::adjoint(const bool add,const  std::shared_ptr<double3DReg> model,  std::shared_ptr<double3DReg> data) const{
	if( not checkDomainRange(model,data)){
		throw std::runtime_error("**** ERROR [staggerX-adjoint]: inconsistent vectors with operator's domain and range ****");
	};
  if(!add) model->scale(0.);

  #pragma omp parallel for collapse(3)
	for(int iy = 0; iy < _ny; iy++){
	  for(int ix = 1; ix < _nx-1; ix++){
	    for(int iz = 0; iz < _nz; iz++){
	      (*model->_mat)[iy][ix][iz] += 0.5 * ((*data->_mat)[iy][ix][iz] + (*data->_mat)[iy][ix-1][iz] );
	    }
	  }
	}

  //handle grid boundary on right side and left side
  #pragma omp parallel for collapse(2)
	for(int iy = 0; iy < _ny; iy++){
	  for(int iz = 0; iz < _nz; iz++){
	    (*model->_mat)[iy][0][iz] += 0.5 * (*data->_mat)[iy][0][iz]; //left
	    (*model->_mat)[iy][_nx-1][iz] += (*data->_mat)[iy][_nx-1][iz] + 0.5 * (*data->_mat)[iy][_nx-2][iz]; //right
	  }
	}
}
/******************************************************************************/

/******************************************************************************/
staggerY::staggerY(const std::shared_ptr<double3DReg> model, const std::shared_ptr<double3DReg> data){
  if( not data->getHyper()->getAxis(1).n == model->getHyper()->getAxis(1).n){
		throw std::runtime_error("**** ERROR [staggerY-init]: inconsistent nz ****");
	}
  if( not data->getHyper()->getAxis(2).n == model->getHyper()->getAxis(2).n){
		throw std::runtime_error("**** ERROR [staggerY-init]: inconsistent nx ****");
	}
	if( not data->getHyper()->getAxis(3).n == model->getHyper()->getAxis(3).n){
		throw std::runtime_error("**** ERROR [staggerY-init]: inconsistent ny ****");
	}

	_ny = model->getHyper()->getAxis(3).n;
  _nx = model->getHyper()->getAxis(2).n;
  _nz = model->getHyper()->getAxis(1).n;

  setDomainRange(model,data);
}

void staggerY::forward(const bool add,const  std::shared_ptr<double3DReg> model,  std::shared_ptr<double3DReg> data) const{
  if( not checkDomainRange(model,data)){
		throw std::runtime_error("**** ERROR [staggerY-forward]: inconsistent vectors with operator's domain and range ****");
	};
  if(!add) data->scale(0.);

  #pragma omp parallel for collapse(3)
	for(int iy = 0; iy < _ny-1; iy++){
	  for(int ix = 0; ix < _nx; ix++){
	    for(int iz = 0; iz < _nz; iz++){
	      (*data->_mat)[iy][ix][iz] += 0.5 * ((*model->_mat)[iy][ix][iz] + (*model->_mat)[iy+1][ix][iz] );
	    }
	  }
	}

  //handle grid boundary on right side
  #pragma omp parallel for collapse(2)
	for(int ix = 0; ix < _nx; ix++){
	  for(int iz = 0; iz < _nz; iz++){
	    (*data->_mat)[_ny-1][ix][iz] += (*model->_mat)[_ny-1][ix][iz];
	  }
	}
}

void staggerY::adjoint(const bool add,const  std::shared_ptr<double3DReg> model,  std::shared_ptr<double3DReg> data) const{
	if( not checkDomainRange(model,data)){
		throw std::runtime_error("**** ERROR [staggerY-adjoint]: inconsistent vectors with operator's domain and range ****");
	};
  if(!add) model->scale(0.);

  #pragma omp parallel for collapse(3)
	for(int iy = 0; iy < _ny-1; iy++){
	  for(int ix = 1; ix < _nx; ix++){
	    for(int iz = 0; iz < _nz; iz++){
	      (*model->_mat)[iy][ix][iz] += 0.5 * ((*data->_mat)[iy][ix][iz] + (*data->_mat)[iy-1][ix][iz] );
	    }
	  }
	}

  //handle grid boundary on right side and left side
  #pragma omp parallel for collapse(2)
	for(int ix = 0; ix < _nx; ix++){
	  for(int iz = 0; iz < _nz; iz++){
	    (*model->_mat)[0][ix][iz] += 0.5 * (*data->_mat)[0][ix][iz]; //left
	    (*model->_mat)[_ny-1][ix][iz] += (*data->_mat)[_ny-1][ix][iz] + 0.5 * (*data->_mat)[_ny-2][ix][iz]; //right
	  }
	}
}
/******************************************************************************/
