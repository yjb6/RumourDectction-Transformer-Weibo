
	$(".animsition").animsition({
	    inClass               :   'fade-in',
	    outClass              :   'fade-out',
	    inDuration            :    800,
	    outDuration           :    1000,
	    linkElement           :   '.animsition-link',

	    loading               :    false,
	    loadingParentElement  :   'body',
	    loadingClass          :   'animsition-loading',
	    unSupportCss          : [ 'animation-duration',
	                              '-webkit-animation-duration',
	                              '-o-animation-duration'
	                            ],


	    overlay               :   false,

	    overlayClass          :   'animsition-overlay-slide',
	    overlayParentElement  :   'body'
  	});