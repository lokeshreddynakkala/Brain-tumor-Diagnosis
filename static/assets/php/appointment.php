<?php
    $to = "info@example.com"; 
    //this is your Email address
    $from = $_POST['email']; 
    //this is the sender's Email address
    //this is firt name
    $first_name = $_POST['name'];
    //this is last name
    $last_name = $_POST['name'];
    //this is subject
    $subject = "Appointment: ";
    //this is message body
    $message = "Appointment: Full Name" . $first_name . " " . $last_name . 
        "\n\n" . " Phone:" . $_POST['phone'] . 
        "\n\n" . " Wrote the following:" . 
        "\n\n" . $_POST['doctor'] . 
        "\n\n" . $_POST['message'];

    $headers = "From:" . $from . "\r\n" .
        'Reply-To: webmaster@example.com' . "\r\n" .
        'X-Mailer: PHP/' . phpversion();

    $headers2 = "From:" . $to;
    mail($to,$subject,$message,$headers);
    mail($from,$subject,$message,$headers); 
?>