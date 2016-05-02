MLProof = {};

MLProof.start = function() {

    MLProof.MODE = 'merge';
    MLProof.LOADED = 0;
    MLProof.LOADED_TARGET = -1;

    // onclick handler for corrections
    $('.correction').on('click', MLProof.apply_correction);

    MLProof.get_error();

    $('#slice_overview img').on('load', MLProof.track_loading);
    $('.borders').on('load', MLProof.track_loading);
    $('.labels').on('load', MLProof.track_loading);

};

MLProof.track_loading = function() {
    MLProof.LOADED += 1;
    
    if (MLProof.LOADED == MLProof.LOADED_TARGET) {
        console.log('All loaded.');
        MLProof.LOADED = 0;
        $('#loading').hide();
    }
};

MLProof.get_error = function() {
  
    console.log('Getting next ' + MLProof.MODE + ' error.');

    $('#loading').show();

    var contents = $.ajax({
            type: "GET",
            url: MLProof.MODE+'/get_correction_count/'+Math.random()

        }).done(function() {

            var response = JSON.parse(contents.responseText);
            correction_count = response['correction_count'];

            MLProof.LOADED_TARGET = 3 + (correction_count)*2;

            // propagate slice overview
            $('#slice_overview img').attr('src', MLProof.MODE+'/slice_overview/0/'+Math.random());
            $('.correction').hide();

            // propagate current
            $('#current .borders').attr('src', MLProof.MODE+'/current/0/'+Math.random());
            $('#current .labels').attr('src', MLProof.MODE+'/current_labels/0/'+Math.random());
            $('#current').show();
            
            // propagate corrections
            
            for (var i=0; i<(correction_count); i++) {
                $('#correction'+(i+1)+' .borders').attr('src', MLProof.MODE+'/correction/'+(i)+'/'+Math.random());
                $('#correction'+(i+1)+' .labels').attr('src', MLProof.MODE+'/correction_labels/'+(i)+'/'+Math.random());

                $('#correction'+(i+1)).show();
            }

        });

};

MLProof.apply_correction = function(e) {


    $('#loading').show();

    var clicked_correction = e.target.parentNode.id;
    if (clicked_correction == 'centered') {
        clicked_correction = e.target.id;
    }
    clicked_correction = clicked_correction.replace('correction', '');

    var contents = $.ajax({
        type: "GET",
        url: MLProof.MODE+'/correct/'+clicked_correction+'/'+Math.random()
    }).done(function() {
        var response = JSON.parse(contents.responseText);
        MLProof.MODE = response['mode'];

        MLProof.get_error();

    });

};