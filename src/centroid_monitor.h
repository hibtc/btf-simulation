// ##################################
// Centroid monitor
// 
// Author: Cristopher Cortés
// Date: 2023-05-03
// ##################################

#ifndef XTRACK_CENTROIDMONITOR_H
#define XTRACK_CENTROIDMONITOR_H

/*gpufun*/
void CentroidMonitor_track_local_particle(CentroidMonitorData el, LocalParticle* part0){

    int64_t const start_at_turn = CentroidMonitorData_get_start_at_turn(el);
    int64_t const stop_at_turn = CentroidMonitorData_get_stop_at_turn(el);
    CentroidRecord record = CentroidMonitorData_getp_data(el);

        int64_t active_parts = 0;
        double x_cen = 0.;
        double y_cen = 0.;
        int64_t at_turn = 0;
//start_per_particle_block (part0->part)
    
    at_turn = LocalParticle_get_at_turn(part);
    
    if (at_turn >= start_at_turn && at_turn <= stop_at_turn){
       
            int64_t part_state = LocalParticle_get_state(part);

            if(part_state > 0){
                x_cen += LocalParticle_get_x(part);
                y_cen += LocalParticle_get_y(part);
                active_parts++;
            }

        //end_per_particle_block
	   
        x_cen = x_cen/active_parts;
        y_cen = y_cen/active_parts;

        int64_t i_slot = at_turn - start_at_turn;
	
        if(i_slot >= 0){
            CentroidRecord_set_x_cen(record, i_slot, x_cen);
            CentroidRecord_set_y_cen(record, i_slot, y_cen);
            CentroidRecord_set_at_turn(record, i_slot, i_slot);   
        }
    }
}

#endif
