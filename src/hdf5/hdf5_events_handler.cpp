#include "hdf5_events_handler.h"

#include "hdf5_utils.h"
#include <algorithm>

HDF5EventsHandler::HDF5EventsHandler()
	: strategy_set(false)
{
	//H5Eset_auto2(H5E_DEFAULT, NULL, NULL);
}

HDF5EventsHandler::~HDF5EventsHandler()
{
}

void
HDF5EventsHandler::beginAction(OperationContext * ctx, hid_t file_id, std::unordered_map < std::string, hid_t > &opened_IDS_files, HDF5Writer & writer, HDF5Reader & reader, std::string & files_directory, std::string & relative_file_path, int access_mode)
{
	//printf("HDF5EventsHandler::beginAction called for context type %d\n", ctx->getType());
	hid_t loc_id = -1;
	strategy_set = false;
	if (ctx->getAccessmode() == WRITE_OP && ctx->getRangemode() == GLOBAL_OP) {
		writer.create_IDS_group(ctx, file_id, opened_IDS_files, files_directory, relative_file_path, access_mode, &loc_id);
		writer.setWriteStrategy(GLOBAL_OP, loc_id);
		strategy_set = true;
	} else if (ctx->getAccessmode() == WRITE_OP && ctx->getRangemode() == SLICE_OP) {
		std::string IDS_link_name = ctx->getDataobjectName();
		std::replace(IDS_link_name.begin(), IDS_link_name.end(), '/', '_');
		HDF5Utils hdf5_utils;
		std::string IDS_pulse_file = hdf5_utils.getIDSPulseFilePath(files_directory, relative_file_path, IDS_link_name);
		bool call_put_required = false;
		if (hdf5_utils.pulseFileExists(IDS_pulse_file)) {
			writer.open_IDS_group(ctx, file_id, opened_IDS_files, files_directory, relative_file_path, &loc_id);
			writer.setWriteStrategy(SLICE_OP, loc_id);
			strategy_set = true;
			if (loc_id == -1) {
				call_put_required = true;
			} 
			else {
				int homogeneous_time = -1;
				writer.read_homogeneous_time(&homogeneous_time, loc_id);
				call_put_required = (homogeneous_time == -1);
				// Group will be closed in endAction() - don't close it here
			}
		}
		else {
			call_put_required = true;
		}
		if (call_put_required) {
			//hid_t loc_id = -1;
			writer.create_IDS_group(ctx, file_id, opened_IDS_files, files_directory, relative_file_path, access_mode, &loc_id);	
			if (!strategy_set) {
				writer.setWriteStrategy(GLOBAL_OP, loc_id);
				strategy_set = true;
			}
		}
		else {
			if (!strategy_set) {
				writer.setWriteStrategy(SLICE_OP, loc_id);
				strategy_set = true;
			}
		}
		
	} else if (ctx->getAccessmode() == READ_OP) {
		//printf("READ_OP detected in beginAction, opening IDS group for reading\n");
		reader.open_IDS_group(ctx, file_id, opened_IDS_files, files_directory, relative_file_path);
		reader.setSliceMode(ctx);
	}
}

void HDF5EventsHandler::endAction(Context * ctx, hid_t file_id, HDF5Writer & writer, HDF5Reader & reader, std::unordered_map < std::string, hid_t > &opened_IDS_files)
{
	//printf("HDF5EventsHandler::endAction called for context type %d\n", ctx->getType());
	if (ctx->getType() == CTX_ARRAYSTRUCT_TYPE) {
		ArraystructContext *aosctx = dynamic_cast<ArraystructContext *>(ctx);
		OperationContext *opCtx = aosctx->getOperationContext();
		if (opCtx->getAccessmode() == WRITE_OP) {
			writer.endAction(ctx);
		} else if (opCtx->getAccessmode() == READ_OP) {
			reader.endAction(ctx);
		}
	} else if (ctx->getType() == CTX_OPERATION_TYPE) {
		OperationContext *opCtx = dynamic_cast < OperationContext * >(ctx);
		if (opCtx->getAccessmode() == WRITE_OP) {
            if (opCtx->getRangemode() == GLOBAL_OP)
			    writer.write_buffers();
			H5Fflush(file_id, H5F_SCOPE_LOCAL);
			writer.endAction(ctx);
			writer.close_datasets();
			writer.close_group(opCtx);
			writer.close_file_handler(opCtx->getDataobjectName(), opened_IDS_files);
			strategy_set = false;
		}
        else if (opCtx->getAccessmode() == READ_OP) {
			reader.endAction(ctx);
			reader.close_datasets();
            reader.close_group(opCtx);
			reader.close_file_handler(opCtx->getDataobjectName(), opened_IDS_files);
			strategy_set = false;
		}
	}
}
