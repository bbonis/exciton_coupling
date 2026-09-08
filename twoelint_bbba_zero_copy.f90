! /twoelint_bbba_zero_copy_v4_correct.f90
!
! Zero-copy Fortran kernel for calc_BBBA (overlap-only path).
! This is a direct port of the logic in:
!   twoelint_calc_ov_binary_2_corr.py : calc_BBBA + calc_term_BBBA_{2,3,4}
!
! Notes:
! - Uses DOUBLE PRECISION (older f2py compatibility; avoids real64 in wrappers).
! - Expects raveled C-order arrays + explicit ncol (nc) for each 2D matrix.
! - Expects pos2/pos3/pos4 to contain 0-based positions into a/b/c/d/v arrays.
!
module twoelint_bbba_mod
  implicit none
  private
  public :: bbba_accum_all, bbba_trace_open, bbba_trace_close

logical, save :: bbba_trace_enabled = .false.
integer, save :: bbba_trace_unit = -1

contains

! ======================================================
! Python-driven trace hooks (NO environment variables)
! ======================================================

subroutine bbba_trace_open(filename)
  character(len=*), intent(in) :: filename
  integer :: ios
  if (bbba_trace_unit > 0) then
    close(bbba_trace_unit)
  end if
  bbba_trace_unit = 991
  open(unit=bbba_trace_unit, file=trim(filename), status="unknown", position="append", action="write", iostat=ios)
  if (ios == 0) then
    bbba_trace_enabled = .true.
  else
    bbba_trace_enabled = .false.
    bbba_trace_unit = -1
  end if
end subroutine bbba_trace_open

subroutine bbba_trace_close()
  if (bbba_trace_unit > 0) then
    close(bbba_trace_unit)
  end if
  bbba_trace_unit = -1
  bbba_trace_enabled = .false.
end subroutine bbba_trace_close

subroutine bbba_trace_event(uc, cx, key, &
                            i_raw, j_raw, k_raw, l_raw, &
                            i_n, j_n, k_n, l_n, &
                            x0, x1, x2, integral, contrib, pref, &
                            lab1, r1, c1, v1, &
                            lab2, r2, c2, v2, &
                            lab3, r3, c3, v3)
  integer, intent(in) :: uc
  character(len=*), intent(in) :: cx, key
  integer, intent(in) :: i_raw, j_raw, k_raw, l_raw
  integer, intent(in) :: i_n, j_n, k_n, l_n
  integer, intent(in) :: x0, x1, x2
  double precision, intent(in) :: integral, contrib, pref
  character(len=*), intent(in) :: lab1, lab2, lab3
  integer, intent(in) :: r1, c1, r2, c2, r3, c3
  double precision, intent(in) :: v1, v2, v3
  integer :: a, b, c, d, t

  if (.not. bbba_trace_enabled) return

  ! canon = sorted(row_norm)
  a = i_n; b = j_n; c = k_n; d = l_n
  if (a > b) then; t=a; a=b; b=t; end if
  if (c > d) then; t=c; c=d; d=t; end if
  if (a > c) then; t=a; a=c; c=t; end if
  if (b > d) then; t=b; b=d; d=t; end if
  if (b > c) then; t=b; b=c; c=t; end if

  write(bbba_trace_unit,'(A,I0,1X,A,"(",I0,", ",I0,", ",I0,", ",I0,")",1X,A,1X,A)') &
    "uc=", uc, "canon=", a, b, c, d, "cx="//trim(cx), "key="//trim(key)

  write(bbba_trace_unit,'(A,"(",I0,", ",I0,", ",I0,", ",I0,")",1X,A,ES24.16,1X,A,ES24.16)') &
    "row=", i_raw, j_raw, k_raw, l_raw, "integral=", integral, "contrib=", contrib

  write(bbba_trace_unit,'(A,"(",I0,", ",I0,", ",I0,", ",I0,")",1X,A,"(",I0,", ",I0,", ",I0,")")') &
    "note=row_norm=", i_n, j_n, k_n, l_n, "BBBA_4 perm=", x0, x1, x2

  write(bbba_trace_unit,'(A,ES24.16)') "  - prefactor=", pref

  if (len_trim(lab1) > 0) then
    write(bbba_trace_unit,'(A,A,"(",I0,", ",I0,")=",ES24.16)') "  - ", trim(lab1), r1, c1, v1
  end if
  if (len_trim(lab2) > 0) then
    write(bbba_trace_unit,'(A,A,"(",I0,", ",I0,")=",ES24.16)') "  - ", trim(lab2), r2, c2, v2
  end if
  if (len_trim(lab3) > 0) then
    write(bbba_trace_unit,'(A,A,"(",I0,", ",I0,")=",ES24.16)') "  - ", trim(lab3), r3, c3, v3
  end if

  flush(bbba_trace_unit)
end subroutine bbba_trace_event


  pure double precision function get2(a, nc, r0, c0) result(x)
    double precision, intent(in) :: a(*)
    integer, intent(in) :: nc
    integer, intent(in) :: r0, c0  ! 0-based
    integer*8 :: idx
    idx = int(r0, 8) * int(nc, 8) + int(c0, 8) + 1_8
    x = a(idx)
  end function get2

  subroutine split_abbb(i, j, k, l, nbas_a, a0, b0, b1, b2, ok)
    integer, intent(in) :: i, j, k, l, nbas_a
    integer, intent(out) :: a0, b0, b1, b2
    logical, intent(out) :: ok

    integer :: na, nb
    integer :: bb(3)
    integer :: idx

    na = 0
    nb = 0

    call push_one(i)
    call push_one(j)
    call push_one(k)
    call push_one(l)

    ok = (na == 1 .and. nb == 3)
    if (.not. ok) then
      a0 = 0; b0 = 0; b1 = 0; b2 = 0
    else
      b0 = bb(1); b1 = bb(2); b2 = bb(3)
    end if

  contains
    subroutine push_one(x)
      integer, intent(in) :: x
      if (x <= nbas_a) then
        na = na + 1
        a0 = x - 1
      else
        nb = nb + 1
        if (nb <= 3) then
          bb(nb) = x - 1 - nbas_a
        end if
      end if
    end subroutine push_one
  end subroutine split_abbb

subroutine bbba_term4( &
    i_raw, j_raw, k_raw, l_raw, &
    i_n, j_n, k_n, l_n, &
    b0, b1, b2, aA, &
    value, term32, term42, term23, term24, &
    sc_a_ba, sc_a_ba_nc, cs_a_ab, cs_a_ab_nc, &
    sc_b_ab, sc_b_ab_nc, cs_b_ba, cs_b_ba_nc, &
    cs_ea_a, cs_ea_a_nc, sc_ip_a, sc_ip_a_nc, &
    aca_b_right, aca_b_right_nc, aca_b_left, aca_b_left_nc, &
    ac_ip_b_right, ac_ip_b_right_nc, ca_ea_b_right, ca_ea_b_right_nc, &
    ac_ip_b_left, ac_ip_b_left_nc, ca_ea_b_left, ca_ea_b_left_nc, &
    t32x, t32c, t42x, t42c, t23x, t23c, t24x, t24c, &
    t31x, t31c, t14x, t14c, t32x_cs, t32c_cs, &
    t24c_2s, t24x_2s, t21c, t21x, t12c, t12x, &
    t34c_ea, t34x_ea, t34c_ip, t34x_ip)
    integer, intent(in) :: i_raw, j_raw, k_raw, l_raw
    integer, intent(in) :: i_n, j_n, k_n, l_n
    integer, intent(in) :: b0, b1, b2, aA
    double precision, intent(in) :: value, term32, term42, term23, term24

    double precision, intent(in) :: sc_a_ba(*), cs_a_ab(*), sc_b_ab(*), cs_b_ba(*)
    double precision, intent(in) :: cs_ea_a(*), sc_ip_a(*)
    double precision, intent(in) :: aca_b_right(*), aca_b_left(*)
    double precision, intent(in) :: ac_ip_b_right(*), ca_ea_b_right(*)
    double precision, intent(in) :: ac_ip_b_left(*), ca_ea_b_left(*)

    integer, intent(in) :: sc_a_ba_nc, cs_a_ab_nc, sc_b_ab_nc, cs_b_ba_nc
    integer, intent(in) :: cs_ea_a_nc, sc_ip_a_nc
    integer, intent(in) :: aca_b_right_nc, aca_b_left_nc
    integer, intent(in) :: ac_ip_b_right_nc, ca_ea_b_right_nc
    integer, intent(in) :: ac_ip_b_left_nc, ca_ea_b_left_nc

    double precision, intent(out) :: t32x, t32c, t42x, t42c, t23x, t23c, t24x, t24c
    double precision, intent(out) :: t31x, t31c, t14x, t14c, t32x_cs, t32c_cs
    double precision, intent(out) :: t24c_2s, t24x_2s, t21c, t21x, t12c, t12x, t34c_ea, t34x_ea, t34c_ip, t34x_ip

    double precision :: Cp, Xp
    double precision :: f1, f2, f3, delta
    integer :: p(6,3)
    integer :: idx

    Cp = 4.0d0
    Xp = -2.0d0

    t32x = 0.0d0; t32c = 0.0d0; t42x = 0.0d0; t42c = 0.0d0
    t23x = 0.0d0; t23c = 0.0d0; t24x = 0.0d0; t24c = 0.0d0
    t32c_cs = 0.0d0; t32x_cs = 0.0d0
    t31c = 0.0d0; t31x = 0.0d0
    t14c = 0.0d0; t14x = 0.0d0
    t24c_2s = 0.0d0; t24x_2s = 0.0d0
    t21c = 0.0d0; t21x = 0.0d0
    t12c = 0.0d0; t12x = 0.0d0
    t34c_ea = 0.0d0; t34x_ea = 0.0d0
    t34c_ip = 0.0d0; t34x_ip = 0.0d0

    ! perms = [p0,p1,p2,p3,p4,p5]
    ! p5=(i0,i1,i2) ; p3=(i0,i2,i1) ; p4=(i1,i0,i2)
    ! p1=(i1,i2,i0) ; p2=(i2,i0,i1) ; p0=(i2,i1,i0)
    p(1,1)=b2; p(1,2)=b1; p(1,3)=b0
    p(2,1)=b1; p(2,2)=b2; p(2,3)=b0
    p(3,1)=b2; p(3,2)=b0; p(3,3)=b1
    p(4,1)=b0; p(4,2)=b2; p(4,3)=b1
    p(5,1)=b1; p(5,2)=b0; p(5,3)=b2
    p(6,1)=b0; p(6,2)=b1; p(6,3)=b2

    ! idx in [0,1] -> Fortran indices [1,2]
    do idx = 1, 2
      call term4_group01(p(idx,1), p(idx,2), p(idx,3))
    end do
    ! idx in [3,5] -> Fortran indices [4,6]
    call term4_group35(p(4,1), p(4,2), p(4,3))
    call term4_group35(p(6,1), p(6,2), p(6,3))
    ! idx in [2,4] -> Fortran indices [3,5]
    call term4_group24(p(3,1), p(3,2), p(3,3))
    call term4_group24(p(5,1), p(5,2), p(5,3))

  contains
    subroutine term4_group01(x0, x1, x2)
      integer, intent(in) :: x0, x1, x2
      double precision :: s1, s2, s3
      double precision :: f1, f2, f3, delta
      t32x_cs = t32x_cs + Xp * get2(sc_b_ab, sc_b_ab_nc, aA, x0) * get2(cs_ea_a, cs_ea_a_nc, 0, x1) * &
                          get2(ac_ip_b_left, ac_ip_b_left_nc, x2, 0) * value

      f1 = get2(cs_b_ba, cs_b_ba_nc, x2, aA)

      f2 = get2(sc_ip_a, sc_ip_a_nc, x1, 0)

      f3 = get2(ca_ea_b_right, ca_ea_b_right_nc, 0, x0)

      delta = Cp * f1 * f2 * f3 * value

      t24c_2s = t24c_2s + delta

      call bbba_trace_event(4, 'C', 'term_24_C_2S_IP', &

                            i_raw, j_raw, k_raw, l_raw, i_n, j_n, k_n, l_n, x0, x1, x2, value, delta, Cp, &

                            'CS_B_BA', x0, aA, f1, 'SC_IP_A', x1, 0, f2, 'CA_EA_B_RIGHT', 0, x2, f3)      
      t14x = t14x + Xp * get2(cs_a_ab, cs_a_ab_nc, aA, x0) * get2(sc_ip_a, sc_ip_a_nc, x2, 0) * &
                    get2(ca_ea_b_right, ca_ea_b_right_nc, 0, x1) * value

      t32c = t32c + Cp * get2(aca_b_right, aca_b_right_nc, x0, x1) * get2(ac_ip_b_left, ac_ip_b_left_nc, x2, 0) * term32
      t23c = t23c + Cp * get2(aca_b_left, aca_b_left_nc, x0, x1) * get2(ac_ip_b_right, ac_ip_b_right_nc, x2, 0) * term23

      t21x = t21x + Xp * get2(sc_a_ba, sc_a_ba_nc, x0, aA) * get2(aca_b_left, aca_b_left_nc, x1, x2) * value

      t34x_ip = t34x_ip + Xp * get2(ac_ip_b_left, ac_ip_b_left_nc, x0, 0) * get2(sc_ip_a, sc_ip_a_nc, x1, 0) * &
                            get2(ca_ea_b_right, ca_ea_b_right_nc, 0, x2) * term32
      t34x_ea = t34x_ea + Xp * get2(ac_ip_b_left, ac_ip_b_left_nc, x2, 0) * get2(cs_ea_a, cs_ea_a_nc, 0, x1)&
        & * get2(ca_ea_b_right, ca_ea_b_right_nc, 0, x0) * term24

      t42x = t42x + Xp * get2(aca_b_right, aca_b_right_nc, x0, x1) * get2(ca_ea_b_left, ca_ea_b_left_nc, 0, x2) * term42
      t24x = t24x + Xp * get2(aca_b_left, aca_b_left_nc, x0, x1) * get2(ca_ea_b_right, ca_ea_b_right_nc, 0, x2) * term24
    end subroutine term4_group01

    subroutine term4_group35(x0, x1, x2)
      integer, intent(in) :: x0, x1, x2

      t24x_2s = t24x_2s + Xp * get2(cs_b_ba, cs_b_ba_nc, x2, aA) * get2(sc_ip_a, sc_ip_a_nc, x1, 0)&
        & * get2(ca_ea_b_right, ca_ea_b_right_nc, 0, x0) * value

      t32c_cs = t32c_cs + Cp * get2(sc_b_ab, sc_b_ab_nc, aA, x0) * get2(cs_ea_a, cs_ea_a_nc, 0, x1) * &
                          get2(ac_ip_b_left, ac_ip_b_left_nc, x2, 0) * value

      t21c = t21c + Cp * get2(sc_a_ba, sc_a_ba_nc, x0, aA) * get2(aca_b_left, aca_b_left_nc, x1, x2) * value

      t12x = t12x + Xp * get2(cs_a_ab, cs_a_ab_nc, aA, x1) * get2(aca_b_right, aca_b_right_nc, x0, x2) * value
      t34c_ip = t34c_ip + Cp * get2(ac_ip_b_left, ac_ip_b_left_nc, x0, 0) * get2(sc_ip_a, sc_ip_a_nc, x1, 0) * &
                            get2(ca_ea_b_right, ca_ea_b_right_nc, 0, x2) * term32
      t31c = t31c + Cp * get2(sc_a_ba, sc_a_ba_nc, x0, aA) * get2(cs_ea_a, cs_ea_a_nc, 0, x1) * &
                    get2(ac_ip_b_left, ac_ip_b_left_nc, x2, 0) * value

      t34c_ea = t34c_ea + Cp * get2(ac_ip_b_left, ac_ip_b_left_nc, x2, 0) * get2(cs_ea_a, cs_ea_a_nc, 0, x1)&
        & * get2(ca_ea_b_right, ca_ea_b_right_nc, 0, x0) * term24

      t42c = t42c + Cp * get2(aca_b_right, aca_b_right_nc, x0, x1) * get2(ca_ea_b_left, ca_ea_b_left_nc, 0, x2) * term42
      t24c = t24c + Cp * get2(aca_b_left, aca_b_left_nc, x0, x1) * get2(ca_ea_b_right, ca_ea_b_right_nc, 0, x2) * term24
    end subroutine term4_group35

    subroutine term4_group24(x0, x1, x2)
      integer, intent(in) :: x0, x1, x2

      t31x = t31x + Xp * get2(sc_a_ba, sc_a_ba_nc, x0, aA) * get2(cs_ea_a, cs_ea_a_nc, 0, x1) * &
                    get2(ac_ip_b_left, ac_ip_b_left_nc, x2, 0) * value

      t14c = t14c + Cp * get2(cs_a_ab, cs_a_ab_nc, aA, x1) * get2(sc_ip_a, sc_ip_a_nc, x0, 0) * &
                    get2(ca_ea_b_right, ca_ea_b_right_nc, 0, x2) * value

      t12c = t12c + Cp * get2(cs_a_ab, cs_a_ab_nc, aA, x1) * get2(aca_b_right, aca_b_right_nc, x0, x2) * value

      t32x = t32x + Xp * get2(aca_b_right, aca_b_right_nc, x0, x1) * get2(ac_ip_b_left, ac_ip_b_left_nc, x2, 0) * term32
      t23x = t23x + Xp * get2(aca_b_left, aca_b_left_nc, x0, x1) * get2(ac_ip_b_right, ac_ip_b_right_nc, x2, 0) * term23
    end subroutine term4_group24
  end subroutine bbba_term4
subroutine bbba_term3( &
    i_raw, j_raw, k_raw, l_raw, &
    i_n, j_n, k_n, l_n, &
    b0, b1, b2, aA, is_type_1, value, term32, term42, term23, term24, &
    sc_a_ba, sc_a_ba_nc, cs_a_ab, cs_a_ab_nc, sc_b_ab, sc_b_ab_nc, cs_b_ba, cs_b_ba_nc, &
    cs_ea_a, cs_ea_a_nc, sc_ip_a, sc_ip_a_nc, &
    aca_b_right, aca_b_right_nc, aca_b_left, aca_b_left_nc, &
    ac_ip_b_right, ac_ip_b_right_nc, ca_ea_b_right, ca_ea_b_right_nc, &
    ac_ip_b_left, ac_ip_b_left_nc, ca_ea_b_left, ca_ea_b_left_nc, &
    t32x, t32c, t42x, t42c, t23x, t23c, t24x, t24c, t31x, t31c, t14x, t14c, t32c_cs, t32x_cs, &
    t24c_2s, t24x_2s, t21c, t21x, t12c, t12x, t34c_ea, t34x_ea, t34c_ip, t34x_ip)
    integer, intent(in) :: i_raw, j_raw, k_raw, l_raw
    integer, intent(in) :: i_n, j_n, k_n, l_n
    integer, intent(in) :: b0, b1, b2, aA
    logical, intent(in) :: is_type_1
    double precision, intent(in) :: value, term32, term42, term23, term24

    double precision, intent(in) :: sc_a_ba(*), cs_a_ab(*), sc_b_ab(*), cs_b_ba(*)
    double precision, intent(in) :: cs_ea_a(*), sc_ip_a(*)
    double precision, intent(in) :: aca_b_right(*), aca_b_left(*)
    double precision, intent(in) :: ac_ip_b_right(*), ca_ea_b_right(*)
    double precision, intent(in) :: ac_ip_b_left(*), ca_ea_b_left(*)

    integer, intent(in) :: sc_a_ba_nc, cs_a_ab_nc, sc_b_ab_nc, cs_b_ba_nc
    integer, intent(in) :: cs_ea_a_nc, sc_ip_a_nc
    integer, intent(in) :: aca_b_right_nc, aca_b_left_nc
    integer, intent(in) :: ac_ip_b_right_nc, ca_ea_b_right_nc
    integer, intent(in) :: ac_ip_b_left_nc, ca_ea_b_left_nc

    double precision, intent(out) :: t32x, t32c, t42x, t42c, t23x, t23c, t24x, t24c
    double precision, intent(out) :: t31x, t31c, t14x, t14c, t32c_cs, t32x_cs
    double precision, intent(out) :: t24c_2s, t24x_2s, t21c, t21x, t12c, t12x, t34c_ea, t34x_ea, t34c_ip, t34x_ip
    double precision :: Cp, Xp
    integer :: p(3,3)
    integer :: a, b, c

    Cp = 4.0d0
    Xp = -2.0d0

    t32x = 0.0d0; t32c = 0.0d0; t42x = 0.0d0; t42c = 0.0d0
    t23x = 0.0d0; t23c = 0.0d0; t24x = 0.0d0; t24c = 0.0d0
    t32c_cs = 0.0d0; t32x_cs = 0.0d0
    t31c = 0.0d0; t31x = 0.0d0
    t14c = 0.0d0; t14x = 0.0d0
    t24c_2s = 0.0d0; t24x_2s = 0.0d0
    t21c = 0.0d0; t21x = 0.0d0
    t12c = 0.0d0; t12x = 0.0d0
    t34c_ea = 0.0d0; t34x_ea = 0.0d0
    t34c_ip = 0.0d0; t34x_ip = 0.0d0

    a = b0; b = b1; c = b2
    ! perms = [p0,p1,p2] with special a==c handling
    p(3,1)=a; p(3,2)=b; p(3,3)=c   ! p2 = (a,b,c)
    p(1,1)=b; p(1,2)=c; p(1,3)=a   ! p0 = (b,c,a)
    p(2,1)=b; p(2,2)=a; p(2,3)=c   ! p1 = (b,a,c)
    if (a == c) then
      ! Python: b, a, c = indexes  (swap a and b)
      a = b1; b = b0; c = b2
      p(1,1)=a; p(1,2)=b; p(1,3)=c   ! p0 = (a,b,c)
      p(3,1)=b; p(3,2)=c; p(3,3)=a   ! p2 = (b,c,a)
      p(2,1)=b; p(2,2)=a; p(2,3)=c   ! p1 = (b,a,c)
    end if

    if (is_type_1) then
      ! Direct statement order matches Python.
      t21x = t21x + Xp * get2(sc_a_ba, sc_a_ba_nc, p(2,1), aA) * get2(aca_b_left, aca_b_left_nc, p(2,2), p(2,3)) * value
      t21c = t21c + Cp * get2(sc_a_ba, sc_a_ba_nc, p(3,1), aA) * get2(aca_b_left, aca_b_left_nc, p(3,2), p(3,3)) * value
      t21x = t21x + Xp * get2(sc_a_ba, sc_a_ba_nc, p(1,1), aA) * get2(aca_b_left, aca_b_left_nc, p(1,2), p(1,3)) * value
      t21c = t21c + Cp * get2(sc_a_ba, sc_a_ba_nc, p(2,1), aA) * get2(aca_b_left, aca_b_left_nc, p(2,2), p(2,3)) * value

      t12x = t12x + Xp * get2(cs_a_ab, cs_a_ab_nc, aA, p(2,2)) * get2(aca_b_right, aca_b_right_nc, p(2,1), p(2,3)) * value
      t12c = t12c + Cp * get2(cs_a_ab, cs_a_ab_nc, aA, p(1,2)) * get2(aca_b_right, aca_b_right_nc, p(1,1), p(1,3)) * value
      t12x = t12x + Xp * get2(cs_a_ab, cs_a_ab_nc, aA, p(3,2)) * get2(aca_b_right, aca_b_right_nc, p(3,1), p(3,3)) * value
      t12c = t12c + Cp * get2(cs_a_ab, cs_a_ab_nc, aA, p(3,2)) * get2(aca_b_right, aca_b_right_nc, p(3,1), p(3,3)) * value

      t14c = t14c + Cp * get2(cs_a_ab, cs_a_ab_nc, aA, p(2,1)) * get2(sc_ip_a, sc_ip_a_nc, p(2,2), 0) * get2(ca_ea_b_right, &
        & ca_ea_b_right_nc, 0, p(2,3)) * value
      t14x = t14x + Xp * get2(cs_a_ab, cs_a_ab_nc, aA, p(3,1)) * get2(sc_ip_a, sc_ip_a_nc, p(3,2), 0) * get2(ca_ea_b_right, &
        & ca_ea_b_right_nc, 0, p(3,3)) * value
      t14c = t14c + Cp * get2(cs_a_ab, cs_a_ab_nc, aA, p(3,1)) * get2(sc_ip_a, sc_ip_a_nc, p(3,2), 0) * get2(ca_ea_b_right, &
        & ca_ea_b_right_nc, 0, p(3,3)) * value
      t14x = t14x + Xp * get2(cs_a_ab, cs_a_ab_nc, aA, p(1,1)) * get2(sc_ip_a, sc_ip_a_nc, p(1,3), 0) * get2(ca_ea_b_right, &
        & ca_ea_b_right_nc, 0, p(1,2)) * value

      t32x_cs = t32x_cs + Xp * get2(sc_b_ab, sc_b_ab_nc, aA, p(2,1)) * get2(cs_ea_a, cs_ea_a_nc, 0, p(2,2)) * get2(ac_ip_b_left, &
        & ac_ip_b_left_nc, p(2,3), 0) * value
      t32c_cs = t32c_cs + Cp * get2(sc_b_ab, sc_b_ab_nc, aA, p(2,1)) * get2(cs_ea_a, cs_ea_a_nc, 0, p(2,2)) * get2(ac_ip_b_left, &
        & ac_ip_b_left_nc, p(2,3), 0) * value
      t32x_cs = t32x_cs + Xp * get2(sc_b_ab, sc_b_ab_nc, aA, p(1,1)) * get2(cs_ea_a, cs_ea_a_nc, 0, p(1,2)) * get2(ac_ip_b_left, &
        & ac_ip_b_left_nc, p(1,3), 0) * value
      t32c_cs = t32c_cs + Cp * get2(sc_b_ab, sc_b_ab_nc, aA, p(3,1)) * get2(cs_ea_a, cs_ea_a_nc, 0, p(3,2)) * get2(ac_ip_b_left, &
        & ac_ip_b_left_nc, p(3,3), 0) * value

      t24x_2s = t24x_2s + Xp * get2(cs_b_ba, cs_b_ba_nc, p(1,1), aA) * get2(sc_ip_a, sc_ip_a_nc, p(1,2), 0)&
        & * get2(ca_ea_b_right, ca_ea_b_right_nc, 0, p(1,3)) * value
      t24c_2s = t24c_2s + Cp * get2(cs_b_ba, cs_b_ba_nc, p(2,1), aA) * get2(sc_ip_a, sc_ip_a_nc, p(2,2), 0)&
        & * get2(ca_ea_b_right, ca_ea_b_right_nc, 0, p(2,3)) * value
      t24x_2s = t24x_2s + Xp * get2(cs_b_ba, cs_b_ba_nc, p(3,1), aA) * get2(sc_ip_a, sc_ip_a_nc, p(3,3), 0)&
        & * get2(ca_ea_b_right, ca_ea_b_right_nc, 0, p(3,2)) * value
      t24c_2s = t24c_2s + Cp * get2(cs_b_ba, cs_b_ba_nc, p(1,3), aA) * get2(sc_ip_a, sc_ip_a_nc, p(1,2), 0)&
        & * get2(ca_ea_b_right, ca_ea_b_right_nc, 0, p(1,1)) * value

      t42x = t42x + Xp * get2(aca_b_right, aca_b_right_nc, p(2,1), p(2,2)) * get2(ca_ea_b_left, ca_ea_b_left_nc, 0, p(2,3)) * term42
      t42c = t42c + Cp * get2(aca_b_right, aca_b_right_nc, p(2,1), p(2,2)) * get2(ca_ea_b_left, ca_ea_b_left_nc, 0, p(2,3)) * term42
      t42x = t42x + Xp * get2(aca_b_right, aca_b_right_nc, p(1,1), p(1,2)) * get2(ca_ea_b_left, ca_ea_b_left_nc, 0, p(1,3)) * term42
      t42c = t42c + Cp * get2(aca_b_right, aca_b_right_nc, p(3,1), p(3,2)) * get2(ca_ea_b_left, ca_ea_b_left_nc, 0, p(3,3)) * term42

      t24x = t24x + Xp * get2(aca_b_left, aca_b_left_nc, p(2,1), p(2,2)) * get2(ca_ea_b_right, ca_ea_b_right_nc, 0, p(2,3)) * term24
      t24c = t24c + Cp * get2(aca_b_left, aca_b_left_nc, p(2,1), p(2,2)) * get2(ca_ea_b_right, ca_ea_b_right_nc, 0, p(2,3)) * term24
      t24x = t24x + Xp * get2(aca_b_left, aca_b_left_nc, p(1,1), p(1,2)) * get2(ca_ea_b_right, ca_ea_b_right_nc, 0, p(1,3)) * term24
      t24c = t24c + Cp * get2(aca_b_left, aca_b_left_nc, p(3,1), p(3,2)) * get2(ca_ea_b_right, ca_ea_b_right_nc, 0, p(3,3)) * term24

      t34c_ea = t34c_ea + Cp * get2(ac_ip_b_left, ac_ip_b_left_nc, p(3,3), 0) * get2(cs_ea_a, cs_ea_a_nc, 0, p(3,2)) * &
        & get2(ca_ea_b_right, ca_ea_b_right_nc, 0, p(3,1)) * term24
      t34x_ea = t34x_ea + Xp * get2(ac_ip_b_left, ac_ip_b_left_nc, p(2,3), 0) * get2(cs_ea_a, cs_ea_a_nc, 0, p(2,2)) * &
        & get2(ca_ea_b_right, ca_ea_b_right_nc, 0, p(2,1)) * term24
      t34c_ea = t34c_ea + Cp * get2(ac_ip_b_left, ac_ip_b_left_nc, p(2,3), 0) * get2(cs_ea_a, cs_ea_a_nc, 0, p(2,2)) * &
        & get2(ca_ea_b_right, ca_ea_b_right_nc, 0, p(2,1)) * term24
      t34x_ea = t34x_ea + Xp * get2(ac_ip_b_left, ac_ip_b_left_nc, p(1,3), 0) * get2(cs_ea_a, cs_ea_a_nc, 0, p(1,2)) * &
        & get2(ca_ea_b_right, ca_ea_b_right_nc, 0, p(1,1)) * term24

      t34x_ip = t34x_ip + Xp * get2(ac_ip_b_left, ac_ip_b_left_nc, p(2,1), 0) * get2(sc_ip_a, sc_ip_a_nc, p(2,2), 0) * &
        & get2(ca_ea_b_right, ca_ea_b_right_nc, 0, p(2,3)) * term32
      t34c_ip = t34c_ip + Cp * get2(ac_ip_b_left, ac_ip_b_left_nc, p(2,1), 0) * get2(sc_ip_a, sc_ip_a_nc, p(2,2), 0) * &
        & get2(ca_ea_b_right, ca_ea_b_right_nc, 0, p(2,3)) * term32
      t34x_ip = t34x_ip + Xp * get2(ac_ip_b_left, ac_ip_b_left_nc, p(1,1), 0) * get2(sc_ip_a, sc_ip_a_nc, p(1,2), 0) * &
        & get2(ca_ea_b_right, ca_ea_b_right_nc, 0, p(1,3)) * term32
      t34c_ip = t34c_ip + Cp * get2(ac_ip_b_left, ac_ip_b_left_nc, p(3,1), 0) * get2(sc_ip_a, sc_ip_a_nc, p(3,2), 0) * &
        & get2(ca_ea_b_right, ca_ea_b_right_nc, 0, p(3,3)) * term32

      t31x = t31x + Xp * get2(sc_a_ba, sc_a_ba_nc, p(2,1), aA) * get2(cs_ea_a, cs_ea_a_nc, 0, p(2,3)) * get2(ac_ip_b_left, &
        & ac_ip_b_left_nc, p(2,2), 0) * value
      t31c = t31c + Cp * get2(sc_a_ba, sc_a_ba_nc, p(2,1), aA) * get2(cs_ea_a, cs_ea_a_nc, 0, p(2,3)) * get2(ac_ip_b_left, &
        & ac_ip_b_left_nc, p(2,2), 0) * value
      t31c = t31c + Cp * get2(sc_a_ba, sc_a_ba_nc, p(3,1), aA) * get2(cs_ea_a, cs_ea_a_nc, 0, p(3,3)) * get2(ac_ip_b_left, &
        & ac_ip_b_left_nc, p(3,2), 0) * value
      t31x = t31x + Xp * get2(sc_a_ba, sc_a_ba_nc, p(1,1), aA) * get2(cs_ea_a, cs_ea_a_nc, 0, p(1,3)) * get2(ac_ip_b_left, &
        & ac_ip_b_left_nc, p(1,2), 0) * value

      t32c = t32c + Cp * get2(aca_b_right, aca_b_right_nc, p(2,1), p(2,2)) * get2(ac_ip_b_left, ac_ip_b_left_nc, p(2,3), 0) * term32
      t32x = t32x + Xp * get2(aca_b_right, aca_b_right_nc, p(3,1), p(3,2)) * get2(ac_ip_b_left, ac_ip_b_left_nc, p(3,3), 0) * term32
      t32c = t32c + Cp * get2(aca_b_right, aca_b_right_nc, p(1,1), p(1,2)) * get2(ac_ip_b_left, ac_ip_b_left_nc, p(1,3), 0) * term32
      t32x = t32x + Xp * get2(aca_b_right, aca_b_right_nc, p(1,1), p(1,2)) * get2(ac_ip_b_left, ac_ip_b_left_nc, p(1,3), 0) * term32

      t23c = t23c + Cp * get2(aca_b_left, aca_b_left_nc, p(2,1), p(2,2)) * get2(ac_ip_b_right, ac_ip_b_right_nc, p(2,3), 0) * term23
      t23x = t23x + Xp * get2(aca_b_left, aca_b_left_nc, p(3,1), p(3,2)) * get2(ac_ip_b_right, ac_ip_b_right_nc, p(3,3), 0) * term23
      t23c = t23c + Cp * get2(aca_b_left, aca_b_left_nc, p(1,1), p(1,2)) * get2(ac_ip_b_right, ac_ip_b_right_nc, p(1,3), 0) * term23
      t23x = t23x + Xp * get2(aca_b_left, aca_b_left_nc, p(1,1), p(1,2)) * get2(ac_ip_b_right, ac_ip_b_right_nc, p(1,3), 0) * term23
    else
      ! else branch in Python
      t31x = t31x + Xp * get2(sc_a_ba, sc_a_ba_nc, p(2,1), aA) * get2(cs_ea_a, cs_ea_a_nc, 0, p(2,2)) * get2(ac_ip_b_left, &
        & ac_ip_b_left_nc, p(2,3), 0) * value
      t31c = t31c + Cp * get2(sc_a_ba, sc_a_ba_nc, p(3,1), aA) * get2(cs_ea_a, cs_ea_a_nc, 0, p(3,2)) * get2(ac_ip_b_left, &
        & ac_ip_b_left_nc, p(3,3), 0) * value

      t34x_ip = t34x_ip + Xp * get2(ac_ip_b_left, ac_ip_b_left_nc, p(1,1), 0) * get2(sc_ip_a, sc_ip_a_nc, p(1,2), 0) * &
        & get2(ca_ea_b_right, ca_ea_b_right_nc, 0, p(1,3)) * term32
      t34c_ip = t34c_ip + Cp * get2(ac_ip_b_left, ac_ip_b_left_nc, p(3,1), 0) * get2(sc_ip_a, sc_ip_a_nc, p(3,2), 0) * &
        & get2(ca_ea_b_right, ca_ea_b_right_nc, 0, p(3,3)) * term32

      t14x = t14x + Xp * get2(cs_a_ab, cs_a_ab_nc, aA, p(2,1)) * get2(sc_ip_a, sc_ip_a_nc, p(2,2), 0) * get2(ca_ea_b_right, &
        & ca_ea_b_right_nc, 0, p(2,3)) * value
      t14c = t14c + Cp * get2(cs_a_ab, cs_a_ab_nc, aA, p(3,1)) * get2(sc_ip_a, sc_ip_a_nc, p(3,2), 0) * get2(ca_ea_b_right, &
        & ca_ea_b_right_nc, 0, p(3,3)) * value

      t24c_2s = t24c_2s + Cp * get2(cs_b_ba, cs_b_ba_nc, p(3,1), aA) * get2(sc_ip_a, sc_ip_a_nc, p(3,2), 0)&
        & * get2(ca_ea_b_right, ca_ea_b_right_nc, 0, p(3,3)) * value
      t24x_2s = t24x_2s + Xp * get2(cs_b_ba, cs_b_ba_nc, p(2,3), aA) * get2(sc_ip_a, sc_ip_a_nc, p(2,1), 0)&
        & * get2(ca_ea_b_right, ca_ea_b_right_nc, 0, p(2,2)) * value

      t32c_cs = t32c_cs + Cp * get2(sc_b_ab, sc_b_ab_nc, aA, p(3,1)) * get2(cs_ea_a, cs_ea_a_nc, 0, p(3,2)) * get2(ac_ip_b_left, &
        & ac_ip_b_left_nc, p(3,3), 0) * value
      t32x_cs = t32x_cs + Xp * get2(sc_b_ab, sc_b_ab_nc, aA, p(1,1)) * get2(cs_ea_a, cs_ea_a_nc, 0, p(1,2)) * get2(ac_ip_b_left, &
        & ac_ip_b_left_nc, p(1,3), 0) * value

      t34x_ea = t34x_ea + Xp * get2(ac_ip_b_left, ac_ip_b_left_nc, p(1,3), 0) * get2(cs_ea_a, cs_ea_a_nc, 0, p(1,2)) * &
        & get2(ca_ea_b_right, ca_ea_b_right_nc, 0, p(1,1)) * term24
      t34c_ea = t34c_ea + Cp * get2(ac_ip_b_left, ac_ip_b_left_nc, p(3,3), 0) * get2(cs_ea_a, cs_ea_a_nc, 0, p(3,2)) * &
        & get2(ca_ea_b_right, ca_ea_b_right_nc, 0, p(3,1)) * term24

      t42x = t42x + Xp * get2(aca_b_right, aca_b_right_nc, p(1,1), p(1,2)) * get2(ca_ea_b_left, ca_ea_b_left_nc, 0, p(1,3)) * term42
      t42c = t42c + Cp * get2(aca_b_right, aca_b_right_nc, p(3,1), p(3,2)) * get2(ca_ea_b_left, ca_ea_b_left_nc, 0, p(3,3)) * term42

      t24x = t24x + Xp * get2(aca_b_left, aca_b_left_nc, p(1,1), p(1,2)) * get2(ca_ea_b_right, ca_ea_b_right_nc, 0, p(1,3)) * term24
      t24c = t24c + Cp * get2(aca_b_left, aca_b_left_nc, p(3,1), p(3,2)) * get2(ca_ea_b_right, ca_ea_b_right_nc, 0, p(3,3)) * term24

      t21x = t21x + Xp * get2(sc_a_ba, sc_a_ba_nc, p(1,1), aA) * get2(aca_b_left, aca_b_left_nc, p(1,2), p(1,3)) * value
      t21c = t21c + Cp * get2(sc_a_ba, sc_a_ba_nc, p(3,1), aA) * get2(aca_b_left, aca_b_left_nc, p(3,2), p(3,3)) * value

      t12x = t12x + Xp * get2(cs_a_ab, cs_a_ab_nc, aA, p(3,2)) * get2(aca_b_right, aca_b_right_nc, p(3,1), p(3,3)) * value
      t12c = t12c + Cp * get2(cs_a_ab, cs_a_ab_nc, aA, p(2,2)) * get2(aca_b_right, aca_b_right_nc, p(2,1), p(2,3)) * value

      t32c = t32c + Cp * get2(aca_b_right, aca_b_right_nc, p(1,1), p(1,2)) * get2(ac_ip_b_left, ac_ip_b_left_nc, p(1,3), 0) * term32
      t32x = t32x + Xp * get2(aca_b_right, aca_b_right_nc, p(2,1), p(2,2)) * get2(ac_ip_b_left, ac_ip_b_left_nc, p(2,3), 0) * term32

      t23c = t23c + Cp * get2(aca_b_left, aca_b_left_nc, p(1,1), p(1,2)) * get2(ac_ip_b_right, ac_ip_b_right_nc, p(1,3), 0) * term23
      t23x = t23x + Xp * get2(aca_b_left, aca_b_left_nc, p(2,1), p(2,2)) * get2(ac_ip_b_right, ac_ip_b_right_nc, p(2,3), 0) * term23
    end if
  end subroutine bbba_term3
subroutine bbba_term2( &
    i_raw, j_raw, k_raw, l_raw, &
    i_n, j_n, k_n, l_n, &
    b0, b1, b2, aA, value, term32, term42, term23, term24, &
    sc_a_ba, sc_a_ba_nc, cs_a_ab, cs_a_ab_nc, sc_b_ab, sc_b_ab_nc, cs_b_ba, cs_b_ba_nc, &
    cs_ea_a, cs_ea_a_nc, sc_ip_a, sc_ip_a_nc, &
    aca_b_right, aca_b_right_nc, aca_b_left, aca_b_left_nc, &
    ac_ip_b_right, ac_ip_b_right_nc, ca_ea_b_right, ca_ea_b_right_nc, &
    ac_ip_b_left, ac_ip_b_left_nc, ca_ea_b_left, ca_ea_b_left_nc, &
    t32c_cs, t32x_cs, t31x, t31c, t14x, t14c, t32x, t32c, t42x, t42c, t23x, t23c, t24x, t24c, &
    t24c_2s, t24x_2s, t21c, t21x, t12c, t12x, t34c_ea, t34x_ea, t34c_ip, t34x_ip)
    integer, intent(in) :: i_raw, j_raw, k_raw, l_raw
    integer, intent(in) :: i_n, j_n, k_n, l_n
    integer, intent(in) :: b0, b1, b2, aA
    double precision, intent(in) :: value, term32, term42, term23, term24

    double precision, intent(in) :: sc_a_ba(*), cs_a_ab(*), sc_b_ab(*), cs_b_ba(*)
    double precision, intent(in) :: cs_ea_a(*), sc_ip_a(*)
    double precision, intent(in) :: aca_b_right(*), aca_b_left(*)
    double precision, intent(in) :: ac_ip_b_right(*), ca_ea_b_right(*)
    double precision, intent(in) :: ac_ip_b_left(*), ca_ea_b_left(*)

    integer, intent(in) :: sc_a_ba_nc, cs_a_ab_nc, sc_b_ab_nc, cs_b_ba_nc
    integer, intent(in) :: cs_ea_a_nc, sc_ip_a_nc
    integer, intent(in) :: aca_b_right_nc, aca_b_left_nc
    integer, intent(in) :: ac_ip_b_right_nc, ca_ea_b_right_nc
    integer, intent(in) :: ac_ip_b_left_nc, ca_ea_b_left_nc

    double precision, intent(out) :: t32c_cs, t32x_cs, t31x, t31c, t14x, t14c
    double precision, intent(out) :: t32x, t32c, t42x, t42c, t23x, t23c, t24x, t24c
    double precision, intent(out) :: t24c_2s, t24x_2s, t21c, t21x, t12c, t12x
    double precision, intent(out) :: t34c_ea, t34x_ea, t34c_ip, t34x_ip
    double precision :: Cp, Xp
    Cp = 4.0d0
    Xp = -2.0d0

    t32c_cs = 0.0d0; t32x_cs = 0.0d0
    t31c = 0.0d0; t31x = 0.0d0
    t14c = 0.0d0; t14x = 0.0d0
    t32x = 0.0d0; t32c = 0.0d0
    t42x = 0.0d0; t42c = 0.0d0
    t23x = 0.0d0; t23c = 0.0d0
    t24x = 0.0d0; t24c = 0.0d0
    t24c_2s = 0.0d0; t24x_2s = 0.0d0
    t21c = 0.0d0; t21x = 0.0d0
    t12c = 0.0d0; t12x = 0.0d0
    t34c_ea = 0.0d0; t34x_ea = 0.0d0
    t34c_ip = 0.0d0; t34x_ip = 0.0d0

    t31c = t31c + Cp * get2(sc_a_ba, sc_a_ba_nc, b0, aA) * get2(cs_ea_a, cs_ea_a_nc, 0, b1) * get2(ac_ip_b_left, &
      & ac_ip_b_left_nc, b2, 0) * value
    t31x = t31x + Xp * get2(sc_a_ba, sc_a_ba_nc, b0, aA) * get2(cs_ea_a, cs_ea_a_nc, 0, b1) * get2(ac_ip_b_left, &
      & ac_ip_b_left_nc, b2, 0) * value

    t21c = t21c + Cp * get2(sc_a_ba, sc_a_ba_nc, b0, aA) * get2(aca_b_left, aca_b_left_nc, b1, b2) * value
    t21x = t21x + Xp * get2(sc_a_ba, sc_a_ba_nc, b0, aA) * get2(aca_b_left, aca_b_left_nc, b1, b2) * value

    t12c = t12c + Cp * get2(cs_a_ab, cs_a_ab_nc, aA, b0) * get2(aca_b_right, aca_b_right_nc, b1, b2) * value
    t12x = t12x + Xp * get2(cs_a_ab, cs_a_ab_nc, aA, b0) * get2(aca_b_right, aca_b_right_nc, b1, b2) * value

    t32c_cs = t32c_cs + Cp * get2(sc_b_ab, sc_b_ab_nc, aA, b0) * get2(cs_ea_a, cs_ea_a_nc, 0, b1) * get2(ac_ip_b_left, &
      & ac_ip_b_left_nc, b2, 0) * value
    t32x_cs = t32x_cs + Xp * get2(sc_b_ab, sc_b_ab_nc, aA, b0) * get2(cs_ea_a, cs_ea_a_nc, 0, b1) * get2(ac_ip_b_left, &
      & ac_ip_b_left_nc, b2, 0) * value

    t14c = t14c + Cp * get2(cs_a_ab, cs_a_ab_nc, aA, b0) * get2(sc_ip_a, sc_ip_a_nc, b1, 0) * get2(ca_ea_b_right, &
      & ca_ea_b_right_nc, 0, b2) * value
    t14x = t14x + Xp * get2(cs_a_ab, cs_a_ab_nc, aA, b0) * get2(sc_ip_a, sc_ip_a_nc, b1, 0) * get2(ca_ea_b_right, &
      & ca_ea_b_right_nc, 0, b2) * value

    t24c_2s = t24c_2s + Cp * get2(cs_b_ba, cs_b_ba_nc, b0, aA) * get2(sc_ip_a, sc_ip_a_nc, b1, 0)&
      & * get2(ca_ea_b_right, ca_ea_b_right_nc, 0, b2) * value

    t24x_2s = t24x_2s + Xp * get2(cs_b_ba, cs_b_ba_nc, b0, aA) * get2(sc_ip_a, sc_ip_a_nc, b1, 0)&
      & * get2(ca_ea_b_right, ca_ea_b_right_nc, 0, b2) * value

    t34c_ip = t34c_ip + Cp * get2(sc_ip_a, sc_ip_a_nc, b0, 0) * get2(ac_ip_b_left, ac_ip_b_left_nc, b1, 0) * get2(ca_ea_b_right, &
      & ca_ea_b_right_nc, 0, b2) * term32
    t34x_ip = t34x_ip + Xp * get2(sc_ip_a, sc_ip_a_nc, b0, 0) * get2(ac_ip_b_left, ac_ip_b_left_nc, b1, 0) * get2(ca_ea_b_right, &
      & ca_ea_b_right_nc, 0, b2) * term32

    t34c_ea = t34c_ea + Cp * get2(cs_ea_a, cs_ea_a_nc, 0, b0) * get2(ca_ea_b_right, ca_ea_b_right_nc, 0, b1) * &
      & get2(ac_ip_b_left, ac_ip_b_left_nc, b2, 0) * term24
    t34x_ea = t34x_ea + Xp * get2(cs_ea_a, cs_ea_a_nc, 0, b0) * get2(ca_ea_b_right, ca_ea_b_right_nc, 0, b1) * &
      & get2(ac_ip_b_left, ac_ip_b_left_nc, b2, 0) * term24

    t42c = t42c + Cp * get2(aca_b_right, aca_b_right_nc, b0, b1) * get2(ca_ea_b_left, ca_ea_b_left_nc, 0, b2) * term42
    t42x = t42x + Xp * get2(aca_b_right, aca_b_right_nc, b0, b1) * get2(ca_ea_b_left, ca_ea_b_left_nc, 0, b2) * term42

    t32c = t32c + Cp * get2(aca_b_right, aca_b_right_nc, b0, b1) * get2(ac_ip_b_left, ac_ip_b_left_nc, b2, 0) * term32
    t32x = t32x + Xp * get2(aca_b_right, aca_b_right_nc, b0, b1) * get2(ac_ip_b_left, ac_ip_b_left_nc, b2, 0) * term32

    t24c = t24c + Cp * get2(aca_b_left, aca_b_left_nc, b0, b1) * get2(ca_ea_b_right, ca_ea_b_right_nc, 0, b2) * term24
    t24x = t24x + Xp * get2(aca_b_left, aca_b_left_nc, b0, b1) * get2(ca_ea_b_right, ca_ea_b_right_nc, 0, b2) * term24

    t23c = t23c + Cp * get2(aca_b_left, aca_b_left_nc, b0, b1) * get2(ac_ip_b_right, ac_ip_b_right_nc, b2, 0) * term23
    t23x = t23x + Xp * get2(aca_b_left, aca_b_left_nc, b0, b1) * get2(ac_ip_b_right, ac_ip_b_right_nc, b2, 0) * term23
  end subroutine bbba_term2

  subroutine bbba_accum_all( &
      pos2, pos3, pos4, &
      a_idx, b_idx, c_idx, d_idx, v_val, &
      nbas_a, &
      sc_a_ba, sc_a_ba_nc, &
      cs_a_ab, cs_a_ab_nc, &
      sc_b_ab, sc_b_ab_nc, &
      cs_b_ba, cs_b_ba_nc, &
      cs_ea_a, cs_ea_a_nc, &
      sc_ip_a, sc_ip_a_nc, &
      sc_ip_b, sc_ip_b_nc, &
      cs_ea_b, cs_ea_b_nc, &
      sc_a_ba_pa, sc_a_ba_pa_nc, &
      cs_a_ab_iq, cs_a_ab_iq_nc, &
      sc_b_ab_pa, sc_b_ab_pa_nc, &
      cs_b_ba_iq, cs_b_ba_iq_nc, &
      cs_ea_a_q, cs_ea_a_q_nc, &
      sc_ip_b_p, sc_ip_b_p_nc, &
      sc_ip_a_p, sc_ip_a_p_nc, &
      cs_ea_b_q, cs_ea_b_q_nc, &
      scs_a_bb_iq, scs_a_bb_iq_nc, &
      scs_a_bb_pa, scs_a_bb_pa_nc, &
      aca_a_right, aca_a_right_nc, &
      aca_b_right, aca_b_right_nc, &
      aca_a_left, aca_a_left_nc, &
      aca_b_left, aca_b_left_nc, &
      ca_ea_a_right, ca_ea_a_right_nc, &
      ac_ip_b_right, ac_ip_b_right_nc, &
      ac_ip_a_right, ac_ip_a_right_nc, &
      ca_ea_b_right, ca_ea_b_right_nc, &
      ca_ea_a_left, ca_ea_a_left_nc, &
      ac_ip_b_left, ac_ip_b_left_nc, &
      ac_ip_a_left, ac_ip_a_left_nc, &
      ca_ea_b_left, ca_ea_b_left_nc, &
      out, err_flag)

    integer*8, intent(in) :: pos2(:), pos3(:), pos4(:)
    integer*2, intent(in) :: a_idx(*), b_idx(*), c_idx(*), d_idx(*)
    double precision, intent(in) :: v_val(*)
    integer, intent(in) :: nbas_a

    double precision, intent(in) :: sc_a_ba(*), cs_a_ab(*), sc_b_ab(*), cs_b_ba(*)
    double precision, intent(in) :: cs_ea_a(*), sc_ip_a(*), sc_ip_b(*), cs_ea_b(*)
    double precision, intent(in) :: sc_a_ba_pa(*), cs_a_ab_iq(*), sc_b_ab_pa(*), cs_b_ba_iq(*)
    double precision, intent(in) :: cs_ea_a_q(*), sc_ip_b_p(*), sc_ip_a_p(*), cs_ea_b_q(*)
    double precision, intent(in) :: scs_a_bb_iq(*), scs_a_bb_pa(*)
    double precision, intent(in) :: aca_a_right(*), aca_b_right(*), aca_a_left(*), aca_b_left(*)
    double precision, intent(in) :: ca_ea_a_right(*), ac_ip_b_right(*), ac_ip_a_right(*), ca_ea_b_right(*)
    double precision, intent(in) :: ca_ea_a_left(*), ac_ip_b_left(*), ac_ip_a_left(*), ca_ea_b_left(*)

    integer, intent(in) :: sc_a_ba_nc, cs_a_ab_nc, sc_b_ab_nc, cs_b_ba_nc
    integer, intent(in) :: cs_ea_a_nc, sc_ip_a_nc, sc_ip_b_nc, cs_ea_b_nc
    integer, intent(in) :: sc_a_ba_pa_nc, cs_a_ab_iq_nc, sc_b_ab_pa_nc, cs_b_ba_iq_nc
    integer, intent(in) :: cs_ea_a_q_nc, sc_ip_b_p_nc, sc_ip_a_p_nc, cs_ea_b_q_nc
    integer, intent(in) :: scs_a_bb_iq_nc, scs_a_bb_pa_nc
    integer, intent(in) :: aca_a_right_nc, aca_b_right_nc, aca_a_left_nc, aca_b_left_nc
    integer, intent(in) :: ca_ea_a_right_nc, ac_ip_b_right_nc, ac_ip_a_right_nc, ca_ea_b_right_nc
    integer, intent(in) :: ca_ea_a_left_nc, ac_ip_b_left_nc, ac_ip_a_left_nc, ca_ea_b_left_nc

    double precision, intent(out) :: out(64)
    integer, intent(out) :: err_flag

    integer*8 :: t, p
    integer :: i, j, k, l
    integer :: i_raw, j_raw, k_raw, l_raw
    double precision :: value
    integer :: aA, b0, b1, b2
    logical :: ok, is_type_1

    double precision :: term32, term42, term23, term24

    ! per-integral term accumulators
    double precision :: t32x, t32c, t42x, t42c, t23x, t23c, t24x, t24c
    double precision :: t31x, t31c, t14x, t14c, t32x_cs, t32c_cs
    double precision :: t24c_2s, t24x_2s, t21c, t21x, t12c, t12x, t34c_ea, t34x_ea, t34c_ip, t34x_ip

    ! totals (same order as Python return)
    double precision :: f32c, f32x, f42c, f42x, f23c, f23x, f24c, f24x
    double precision :: f31c, f31x, f14c, f14x, f32c_cs_tot, f32x_cs_tot
    double precision :: f24c_2s_tot, f24x_2s_tot, f21c_tot, f21x_tot, f12c_tot, f12x_tot
    double precision :: f34c_ea_tot, f34x_ea_tot, f34c_ip_tot, f34x_ip_tot

    err_flag = 0
    out = 0.0d0
    f32c = 0.0d0; f32x = 0.0d0; f42c = 0.0d0; f42x = 0.0d0
    f23c = 0.0d0; f23x = 0.0d0; f24c = 0.0d0; f24x = 0.0d0
    f31c = 0.0d0; f31x = 0.0d0; f14c = 0.0d0; f14x = 0.0d0
    f32c_cs_tot = 0.0d0; f32x_cs_tot = 0.0d0
    f24c_2s_tot = 0.0d0; f24x_2s_tot = 0.0d0
    f21c_tot = 0.0d0; f21x_tot = 0.0d0
    f12c_tot = 0.0d0; f12x_tot = 0.0d0
    f34c_ea_tot = 0.0d0; f34x_ea_tot = 0.0d0
    f34c_ip_tot = 0.0d0; f34x_ip_tot = 0.0d0

    ! -------------------------
    ! unique_count == 4 block
    ! -------------------------
    do t = 1_8, int(size(pos4), 8)
      p = pos4(t) + 1_8
      i = int(a_idx(p))
      j = int(b_idx(p))
      k = int(c_idx(p))
      l = int(d_idx(p))
      i_raw = i; j_raw = j; k_raw = k; l_raw = l
      value = v_val(p)

      ! normalize by NBAS_A membership
      if (l <= nbas_a) then
        call rot_l_in_a_uc4(i, j, k, l)
      else if (j <= nbas_a) then
        call swap_ij(i, j)
      else
        err_flag = 1
        cycle
      end if

      ! compact j,k,l rule
      if ((j >= k) .and. (k < l)) then
        err_flag = 1
        cycle
      end if
      if ((j < k) .and. (j <= l)) then
        call swap_kl(k, l)
      end if

      call split_abbb(i, j, k, l, nbas_a, aA, b0, b1, b2, ok)
      if (.not. ok) then
        err_flag = 1
        cycle
      end if

      term32 = value * get2(ca_ea_a_left, ca_ea_a_left_nc, 0, aA)
      term42 = value * get2(ac_ip_a_left, ac_ip_a_left_nc, aA, 0)
      term23 = value * get2(ca_ea_a_right, ca_ea_a_right_nc, 0, aA)
      term24 = value * get2(ac_ip_a_right, ac_ip_a_right_nc, aA, 0)

call bbba_term4( &
    i_raw, j_raw, k_raw, l_raw, &
    i, j, k, l, &
    b0, b1, b2, aA, &
    value, term32, term42, term23, term24, &
    sc_a_ba, sc_a_ba_nc, cs_a_ab, cs_a_ab_nc, &
    sc_b_ab, sc_b_ab_nc, cs_b_ba, cs_b_ba_nc, &
    cs_ea_a, cs_ea_a_nc, sc_ip_a, sc_ip_a_nc, &
    aca_b_right, aca_b_right_nc, aca_b_left, aca_b_left_nc, &
    ac_ip_b_right, ac_ip_b_right_nc, ca_ea_b_right, ca_ea_b_right_nc, &
    ac_ip_b_left, ac_ip_b_left_nc, ca_ea_b_left, ca_ea_b_left_nc, &
    t32x, t32c, t42x, t42c, t23x, t23c, t24x, t24c, &
    t31x, t31c, t14x, t14c, t32x_cs, t32c_cs, &
    t24c_2s, t24x_2s, t21c, t21x, t12c, t12x, &
    t34c_ea, t34x_ea, t34c_ip, t34x_ip)

      call accum_new_u4(b0, b1, b2, aA, value)
      call add_totals()
    end do

    ! -------------------------
    ! unique_count == 3 block
    ! -------------------------
    do t = 1_8, int(size(pos3), 8)
      p = pos3(t) + 1_8
      i = int(a_idx(p))
      j = int(b_idx(p))
      k = int(c_idx(p))
      l = int(d_idx(p))
      i_raw = i; j_raw = j; k_raw = k; l_raw = l
      value = v_val(p)

      if (l <= nbas_a) then
        call rot_l_in_a_uc3(i, j, k, l)
      else if (j <= nbas_a) then
        ! Python leaves (i,j) unchanged for unique_count==3
      else
        err_flag = 1
        cycle
      end if

      is_type_1 = (k /= l)
      if (j == k) then
        call swap_kl(k, l)
      end if

      call split_abbb(i, j, k, l, nbas_a, aA, b0, b1, b2, ok)
      if (.not. ok) then
        err_flag = 1
        cycle
      end if

      term32 = value * get2(ca_ea_a_left, ca_ea_a_left_nc, 0, aA)
      term42 = value * get2(ac_ip_a_left, ac_ip_a_left_nc, aA, 0)
      term23 = value * get2(ca_ea_a_right, ca_ea_a_right_nc, 0, aA)
      term24 = value * get2(ac_ip_a_right, ac_ip_a_right_nc, aA, 0)

      call bbba_term3(i_raw, j_raw, k_raw, l_raw, i, j, k, l, b0, b1, b2, aA, is_type_1, value, term32, term42, term23, term24, &
                      sc_a_ba, sc_a_ba_nc, cs_a_ab, cs_a_ab_nc, sc_b_ab, sc_b_ab_nc, cs_b_ba, cs_b_ba_nc, &
                      cs_ea_a, cs_ea_a_nc, sc_ip_a, sc_ip_a_nc, &
                      aca_b_right, aca_b_right_nc, aca_b_left, aca_b_left_nc, &
                      ac_ip_b_right, ac_ip_b_right_nc, ca_ea_b_right, ca_ea_b_right_nc, &
                      ac_ip_b_left, ac_ip_b_left_nc, ca_ea_b_left, ca_ea_b_left_nc, &
                      t32x, t32c, t42x, t42c, t23x, t23c, t24x, t24c, t31x, t31c, t14x, t14c, t32c_cs, t32x_cs, &
                      t24c_2s, t24x_2s, t21c, t21x, t12c, t12x, t34c_ea, t34x_ea, t34c_ip, t34x_ip)

      call accum_new_u3(b0, b1, b2, aA, value, is_type_1)
      call add_totals()
    end do

    ! -------------------------
    ! unique_count == 2 block
    ! -------------------------
    do t = 1_8, int(size(pos2), 8)
      p = pos2(t) + 1_8
      i = int(a_idx(p))
      j = int(b_idx(p))
      k = int(c_idx(p))
      l = int(d_idx(p))
      i_raw = i; j_raw = j; k_raw = k; l_raw = l
      value = v_val(p)

      call split_abbb(i, j, k, l, nbas_a, aA, b0, b1, b2, ok)
      if (.not. ok) then
        err_flag = 1
        cycle
      end if

      term32 = value * get2(ca_ea_a_left, ca_ea_a_left_nc, 0, aA)
      term42 = value * get2(ac_ip_a_left, ac_ip_a_left_nc, aA, 0)
      term23 = value * get2(ca_ea_a_right, ca_ea_a_right_nc, 0, aA)
      term24 = value * get2(ac_ip_a_right, ac_ip_a_right_nc, aA, 0)

      call bbba_term2(i_raw, j_raw, k_raw, l_raw, i, j, k, l, b0, b1, b2, aA, value, term32, term42, term23, term24, &
                      sc_a_ba, sc_a_ba_nc, cs_a_ab, cs_a_ab_nc, sc_b_ab, sc_b_ab_nc, cs_b_ba, cs_b_ba_nc, &
                      cs_ea_a, cs_ea_a_nc, sc_ip_a, sc_ip_a_nc, &
                      aca_b_right, aca_b_right_nc, aca_b_left, aca_b_left_nc, &
                      ac_ip_b_right, ac_ip_b_right_nc, ca_ea_b_right, ca_ea_b_right_nc, &
                      ac_ip_b_left, ac_ip_b_left_nc, ca_ea_b_left, ca_ea_b_left_nc, &
                      t32c_cs, t32x_cs, t31x, t31c, t14x, t14c, t32x, t32c, t42x, t42c, t23x, t23c, t24x, t24c, &
                      t24c_2s, t24x_2s, t21c, t21x, t12c, t12x, t34c_ea, t34x_ea, t34c_ip, t34x_ip)

      call accum_new_u2(b0, b1, b2, aA, value)
      call add_totals_term2()
    end do

    ! pack output in exact Python return order
    out(1)  = f32c
    out(2)  = f32x
    out(3)  = f42c
    out(4)  = f42x
    out(5)  = f23c
    out(6)  = f23x
    out(7)  = f24c
    out(8)  = f24x
    out(9)  = f31c
    out(10) = f31x
    out(11) = f14c
    out(12) = f14x
    out(13) = f32c_cs_tot
    out(14) = f32x_cs_tot
    out(15) = f24c_2s_tot
    out(16) = f24x_2s_tot
    out(17) = f21c_tot
    out(18) = f21x_tot
    out(19) = f12c_tot
    out(20) = f12x_tot
    out(21) = f34c_ea_tot
    out(22) = f34x_ea_tot
    out(23) = f34c_ip_tot
    out(24) = f34x_ip_tot

  contains
    ! New BBBA terms (outputs 25:64).  The Python expressions use singleton
    ! state axes; these helpers write the equivalent contractions explicitly
    ! so the Fortran kernel also remains valid when that axis is larger than 1.
    double precision function g25(x0, x1, x2, ia, val)
      integer, intent(in) :: x0, x1, x2, ia
      double precision, intent(in) :: val
      integer :: q
      g25 = 0.0d0
      do q = 0, sc_ip_a_p_nc - 1
        g25 = g25 + get2(cs_ea_a_q, cs_ea_a_q_nc, q, x0) &
             & * get2(ac_ip_b_right, ac_ip_b_right_nc, x1, q) &
             & * get2(ca_ea_b_left, ca_ea_b_left_nc, q, x2) &
             & * get2(ac_ip_a_left, ac_ip_a_left_nc, ia, q)
      end do
      g25 = g25 * val
    end function g25

    double precision function g27(x0, x1, x2, ia, val)
      integer, intent(in) :: x0, x1, x2, ia
      double precision, intent(in) :: val
      integer :: q
      g27 = 0.0d0
      do q = 0, sc_ip_a_p_nc - 1
        g27 = g27 + get2(sc_ip_a_p, sc_ip_a_p_nc, x0, q) &
             & * get2(ca_ea_b_left, ca_ea_b_left_nc, q, x1) &
             & * get2(ac_ip_b_right, ac_ip_b_right_nc, x2, q) &
             & * get2(ca_ea_a_right, ca_ea_a_right_nc, q, ia)
      end do
      g27 = g27 * val
    end function g27

    double precision function g29(x0, x1, x2, ia, val)
      integer, intent(in) :: x0, x1, x2, ia
      double precision, intent(in) :: val
      g29 = get2(aca_b_left, aca_b_left_nc, x0, x1) &
           & * get2(cs_a_ab_iq, cs_a_ab_iq_nc, ia, x2) * val
    end function g29

    double precision function g31(x0, x1, x2, ia, val)
      integer, intent(in) :: x0, x1, x2, ia
      double precision, intent(in) :: val
      integer :: q
      g31 = 0.0d0
      do q = 0, sc_ip_a_p_nc - 1
        g31 = g31 + get2(ac_ip_b_left, ac_ip_b_left_nc, x2, q) &
             & * get2(ca_ea_a_left, ca_ea_a_left_nc, q, ia)
      end do
      g31 = get2(scs_a_bb_iq, scs_a_bb_iq_nc, x0, x1) * g31 * val
    end function g31

    double precision function g33(x0, x1, x2, ia, val)
      integer, intent(in) :: x0, x1, x2, ia
      double precision, intent(in) :: val
      integer :: q
      g33 = 0.0d0
      do q = 0, sc_ip_a_p_nc - 1
        g33 = g33 + get2(ac_ip_b_right, ac_ip_b_right_nc, x0, q) &
             & * get2(cs_ea_a_q, cs_ea_a_q_nc, q, x1)
      end do
      g33 = g33 * get2(sc_a_ba_pa, sc_a_ba_pa_nc, x2, ia) * val
    end function g33

    double precision function g35(x0, x1, x2, ia, val)
      integer, intent(in) :: x0, x1, x2, ia
      double precision, intent(in) :: val
      integer :: q
      g35 = 0.0d0
      do q = 0, sc_ip_a_p_nc - 1
        g35 = g35 + get2(sc_ip_a, sc_ip_a_nc, x0, q) &
             & * get2(ca_ea_b_right, ca_ea_b_right_nc, q, x1)
      end do
      g35 = g35 * get2(sc_a_ba_pa, sc_a_ba_pa_nc, x2, ia) * val
    end function g35

    double precision function g37(x0, x1, x2, ia, val)
      integer, intent(in) :: x0, x1, x2, ia
      double precision, intent(in) :: val
      g37 = get2(scs_a_bb_iq, scs_a_bb_iq_nc, x0, x1) &
           & * get2(cs_b_ba, cs_b_ba_nc, x2, ia) * val
    end function g37

    double precision function g39(x0, x1, x2, ia, val)
      integer, intent(in) :: x0, x1, x2, ia
      double precision, intent(in) :: val
      integer :: q
      g39 = 0.0d0
      do q = 0, sc_ip_a_p_nc - 1
        g39 = g39 + get2(ac_ip_b_left, ac_ip_b_left_nc, x0, q) &
             & * get2(cs_ea_a, cs_ea_a_nc, q, x1)
      end do
      g39 = g39 * get2(cs_a_ab_iq, cs_a_ab_iq_nc, ia, x2) * val
    end function g39

    double precision function g41(x0, x1, x2, ia, val)
      integer, intent(in) :: x0, x1, x2, ia
      double precision, intent(in) :: val
      integer :: q
      g41 = 0.0d0
      do q = 0, sc_ip_a_p_nc - 1
        g41 = g41 + get2(cs_ea_a_q, cs_ea_a_q_nc, q, x0) &
             & * get2(sc_ip_b, sc_ip_b_nc, ia, q)
      end do
      g41 = g41 * get2(scs_a_bb_pa, scs_a_bb_pa_nc, x1, x2) * val
    end function g41

    double precision function g43(x0, x1, x2, ia, val)
      integer, intent(in) :: x0, x1, x2, ia
      double precision, intent(in) :: val
      integer :: q
      g43 = 0.0d0
      do q = 0, sc_ip_a_p_nc - 1
        g43 = g43 + get2(ca_ea_b_right, ca_ea_b_right_nc, q, x2) &
             & * get2(ac_ip_a_right, ac_ip_a_right_nc, ia, q)
      end do
      g43 = get2(scs_a_bb_pa, scs_a_bb_pa_nc, x0, x1) * g43 * val
    end function g43

    double precision function g45(x0, x1, x2, ia, val)
      integer, intent(in) :: x0, x1, x2, ia
      double precision, intent(in) :: val
      integer :: q
      g45 = 0.0d0
      do q = 0, sc_ip_a_p_nc - 1
        g45 = g45 + get2(ca_ea_b_left, ca_ea_b_left_nc, q, x0) &
             & * get2(cs_ea_a_q, cs_ea_a_q_nc, q, x1) &
             & * get2(sc_ip_a_p, sc_ip_a_p_nc, x2, q) &
             & * get2(sc_ip_b, sc_ip_b_nc, ia, q)
      end do
      g45 = g45 * val
    end function g45

    double precision function g47(x0, x1, x2, ia, val)
      integer, intent(in) :: x0, x1, x2, ia
      double precision, intent(in) :: val
      integer :: q
      g47 = 0.0d0
      do q = 0, sc_ip_a_p_nc - 1
        g47 = g47 + get2(cs_ea_a, cs_ea_a_nc, q, x2) &
             & * get2(sc_ip_b_p, sc_ip_b_p_nc, ia, q)
      end do
      g47 = get2(scs_a_bb_iq, scs_a_bb_iq_nc, x0, x1) * g47 * val
    end function g47

    double precision function g49(x0, x1, x2, ia, val)
      integer, intent(in) :: x0, x1, x2, ia
      double precision, intent(in) :: val
      integer :: q
      g49 = 0.0d0
      do q = 0, sc_ip_a_p_nc - 1
        g49 = g49 + get2(ac_ip_b_right, ac_ip_b_right_nc, x0, q) &
             & * get2(cs_ea_a_q, cs_ea_a_q_nc, q, x1)
      end do
      g49 = g49 * get2(cs_a_ab, cs_a_ab_nc, ia, x2) * val
    end function g49

    double precision function g51(x0, x1, x2, ia, val)
      integer, intent(in) :: x0, x1, x2, ia
      double precision, intent(in) :: val
      integer :: q
      g51 = 0.0d0
      do q = 0, sc_ip_a_p_nc - 1
        g51 = g51 + get2(ca_ea_b_left, ca_ea_b_left_nc, q, x2) &
             & * get2(ac_ip_a_left, ac_ip_a_left_nc, ia, q)
      end do
      g51 = get2(scs_a_bb_iq, scs_a_bb_iq_nc, x0, x1) * g51 * val
    end function g51

    double precision function g53(x0, x1, x2, ia, val)
      integer, intent(in) :: x0, x1, x2, ia
      double precision, intent(in) :: val
      g53 = get2(scs_a_bb_iq, scs_a_bb_iq_nc, x0, x1) &
           & * get2(sc_b_ab_pa, sc_b_ab_pa_nc, ia, x2) * val
    end function g53

    double precision function g55(x0, x1, x2, ia, val)
      integer, intent(in) :: x0, x1, x2, ia
      double precision, intent(in) :: val
      integer :: q
      g55 = 0.0d0
      do q = 0, sc_ip_a_p_nc - 1
        g55 = g55 + get2(ac_ip_b_right, ac_ip_b_right_nc, x2, q) &
             & * get2(ca_ea_a_right, ca_ea_a_right_nc, q, ia)
      end do
      g55 = get2(scs_a_bb_pa, scs_a_bb_pa_nc, x0, x1) * g55 * val
    end function g55

    double precision function g57(x0, x1, x2, ia, val)
      integer, intent(in) :: x0, x1, x2, ia
      double precision, intent(in) :: val
      integer :: q
      g57 = 0.0d0
      do q = 0, sc_ip_a_p_nc - 1
        g57 = g57 + get2(sc_ip_a_p, sc_ip_a_p_nc, x0, q) &
             & * get2(ca_ea_b_left, ca_ea_b_left_nc, q, x1)
      end do
      g57 = g57 * get2(sc_a_ba, sc_a_ba_nc, x2, ia) * val
    end function g57

    double precision function g59(x0, x1, x2, ia, val)
      integer, intent(in) :: x0, x1, x2, ia
      double precision, intent(in) :: val
      integer :: q
      g59 = 0.0d0
      do q = 0, sc_ip_a_p_nc - 1
        g59 = g59 + get2(sc_ip_a, sc_ip_a_nc, x2, q) &
             & * get2(cs_ea_b_q, cs_ea_b_q_nc, q, ia)
      end do
      g59 = get2(scs_a_bb_pa, scs_a_bb_pa_nc, x0, x1) * g59 * val
    end function g59

    double precision function g61(x0, x1, x2, ia, val)
      integer, intent(in) :: x0, x1, x2, ia
      double precision, intent(in) :: val
      integer :: q
      g61 = 0.0d0
      do q = 0, sc_ip_a_p_nc - 1
        g61 = g61 + get2(sc_ip_a_p, sc_ip_a_p_nc, x2, q) &
             & * get2(cs_ea_b, cs_ea_b_nc, q, ia)
      end do
      g61 = get2(scs_a_bb_iq, scs_a_bb_iq_nc, x0, x1) * g61 * val
    end function g61

    double precision function g63(x0, x1, x2, ia, val)
      integer, intent(in) :: x0, x1, x2, ia
      double precision, intent(in) :: val
      integer :: q
      g63 = 0.0d0
      do q = 0, sc_ip_a_p_nc - 1
        g63 = g63 + get2(ac_ip_b_right, ac_ip_b_right_nc, x0, q) &
             & * get2(cs_ea_a_q, cs_ea_a_q_nc, q, x1) &
             & * get2(sc_ip_a_p, sc_ip_a_p_nc, x2, q) &
             & * get2(cs_ea_b, cs_ea_b_nc, q, ia)
      end do
      g63 = g63 * val
    end function g63

    subroutine add_new_c(slot, base)
      integer, intent(in) :: slot
      double precision, intent(in) :: base
      out(slot) = out(slot) + 4.0d0 * base
    end subroutine add_new_c

    subroutine add_new_x(slot, base)
      integer, intent(in) :: slot
      double precision, intent(in) :: base
      out(slot + 1) = out(slot + 1) - 2.0d0 * base
    end subroutine add_new_x

    subroutine add_new_perm(slot, is_c, x0, x1, x2, ia, val)
      integer, intent(in) :: slot, x0, x1, x2, ia
      logical, intent(in) :: is_c
      double precision, intent(in) :: val
      double precision :: base
      select case (slot)
      case (25); base = g25(x0, x1, x2, ia, val)
      case (27); base = g27(x0, x1, x2, ia, val)
      case (29); base = g29(x0, x1, x2, ia, val)
      case (31); base = g31(x0, x1, x2, ia, val)
      case (33); base = g33(x0, x1, x2, ia, val)
      case (35); base = g35(x0, x1, x2, ia, val)
      case (37); base = g37(x0, x1, x2, ia, val)
      case (39); base = g39(x0, x1, x2, ia, val)
      case (41); base = g41(x0, x1, x2, ia, val)
      case (43); base = g43(x0, x1, x2, ia, val)
      case (45); base = g45(x0, x1, x2, ia, val)
      case (47); base = g47(x0, x1, x2, ia, val)
      case (49); base = g49(x0, x1, x2, ia, val)
      case (51); base = g51(x0, x1, x2, ia, val)
      case (53); base = g53(x0, x1, x2, ia, val)
      case (55); base = g55(x0, x1, x2, ia, val)
      case (57); base = g57(x0, x1, x2, ia, val)
      case (59); base = g59(x0, x1, x2, ia, val)
      case (61); base = g61(x0, x1, x2, ia, val)
      case (63); base = g63(x0, x1, x2, ia, val)
      case default
        base = 0.0d0
      end select
      if (is_c) then
        call add_new_c(slot, base)
      else
        call add_new_x(slot, base)
      end if
    end subroutine add_new_perm

    subroutine add_new_pair(slot, x0, x1, x2, ia, val)
      integer, intent(in) :: slot, x0, x1, x2, ia
      double precision, intent(in) :: val
      call add_new_perm(slot, .true.,  x0, x1, x2, ia, val)
      call add_new_perm(slot, .false., x0, x1, x2, ia, val)
    end subroutine add_new_pair

    subroutine accum_new_u2(a0, a1, a2, ia, val)
      integer, intent(in) :: a0, a1, a2, ia
      double precision, intent(in) :: val
      ! Reorder arguments where the unique-count-2 Python expression places
      ! the singled-out SC/CS factor on the first rather than third B index.
      call add_new_pair(25, a0, a2, a1, ia, val)
      call add_new_pair(27, a0, a1, a2, ia, val)
      call add_new_pair(29, a1, a2, a0, ia, val)
      call add_new_pair(31, a0, a1, a2, ia, val)
      call add_new_pair(33, a2, a1, a0, ia, val)
      call add_new_pair(35, a2, a1, a0, ia, val)
      call add_new_pair(37, a0, a1, a2, ia, val)
      call add_new_pair(39, a2, a1, a0, ia, val)
      call add_new_pair(41, a2, a0, a1, ia, val)
      call add_new_pair(43, a0, a1, a2, ia, val)
      call add_new_pair(45, a0, a1, a2, ia, val)
      call add_new_pair(47, a0, a1, a2, ia, val)
      call add_new_pair(49, a2, a1, a0, ia, val)
      call add_new_pair(51, a0, a1, a2, ia, val)
      call add_new_pair(53, a0, a1, a2, ia, val)
      call add_new_pair(55, a0, a1, a2, ia, val)
      call add_new_pair(57, a2, a1, a0, ia, val)
      call add_new_pair(59, a0, a1, a2, ia, val)
      call add_new_pair(61, a0, a1, a2, ia, val)
      call add_new_pair(63, a1, a2, a0, ia, val)
    end subroutine accum_new_u2

    subroutine accum_new_u3(a0, a1, a2, ia, val, is_t1)
      integer, intent(in) :: a0, a1, a2, ia
      double precision, intent(in) :: val
      logical, intent(in) :: is_t1
      integer :: pnew(3,3), aa, bb, cc

      aa=a0; bb=a1; cc=a2
      pnew(3,:) = (/ aa, bb, cc /)
      pnew(1,:) = (/ bb, cc, aa /)
      pnew(2,:) = (/ bb, aa, cc /)
      if (aa == cc) then
        aa=a1; bb=a0; cc=a2
        pnew(1,:) = (/ aa, bb, cc /)
        pnew(3,:) = (/ bb, cc, aa /)
        pnew(2,:) = (/ bb, aa, cc /)
      end if

      if (is_t1) then
        call add_new_perm(31,.true., pnew(2,1),pnew(2,2),pnew(2,3),ia,val)
        call add_new_perm(31,.true., pnew(1,1),pnew(1,2),pnew(1,3),ia,val)
        call add_new_perm(31,.false.,pnew(3,1),pnew(3,2),pnew(3,3),ia,val)
        call add_new_perm(31,.false.,pnew(1,1),pnew(1,2),pnew(1,3),ia,val)
        call add_new_perm(33,.true., pnew(2,1),pnew(2,2),pnew(2,3),ia,val)
        call add_new_perm(33,.true., pnew(1,1),pnew(1,2),pnew(1,3),ia,val)
        call add_new_perm(33,.false.,pnew(3,1),pnew(3,2),pnew(3,3),ia,val)
        call add_new_perm(33,.false.,pnew(1,1),pnew(1,2),pnew(1,3),ia,val)
        call add_new_perm(35,.true., pnew(2,1),pnew(2,2),pnew(2,3),ia,val)
        call add_new_perm(35,.true., pnew(1,1),pnew(1,2),pnew(1,3),ia,val)
        call add_new_perm(35,.false.,pnew(3,1),pnew(3,2),pnew(3,3),ia,val)
        call add_new_perm(35,.false.,pnew(1,1),pnew(1,2),pnew(1,3),ia,val)
        call add_new_perm(37,.true., pnew(2,1),pnew(2,2),pnew(2,3),ia,val)
        call add_new_perm(37,.true., pnew(1,1),pnew(1,2),pnew(1,3),ia,val)
        call add_new_perm(37,.false.,pnew(3,1),pnew(3,2),pnew(3,3),ia,val)
        call add_new_perm(37,.false.,pnew(1,1),pnew(1,2),pnew(1,3),ia,val)

        call add_new_perm(39,.true., pnew(3,1),pnew(3,2),pnew(3,3),ia,val)
        call add_new_perm(39,.true., pnew(2,1),pnew(2,2),pnew(2,3),ia,val)
        call add_new_perm(39,.false.,pnew(2,1),pnew(2,2),pnew(2,3),ia,val)
        call add_new_perm(39,.false.,pnew(1,1),pnew(1,2),pnew(1,3),ia,val)
        call add_new_perm(43,.true., pnew(3,1),pnew(3,2),pnew(3,3),ia,val)
        call add_new_perm(43,.true., pnew(2,1),pnew(2,2),pnew(2,3),ia,val)
        call add_new_perm(43,.false.,pnew(2,1),pnew(2,2),pnew(2,3),ia,val)
        call add_new_perm(43,.false.,pnew(1,1),pnew(1,2),pnew(1,3),ia,val)
        call add_new_perm(49,.true., pnew(3,1),pnew(3,2),pnew(3,3),ia,val)
        call add_new_perm(49,.true., pnew(2,1),pnew(2,2),pnew(2,3),ia,val)
        call add_new_perm(49,.false.,pnew(2,1),pnew(2,2),pnew(2,3),ia,val)
        call add_new_perm(49,.false.,pnew(1,1),pnew(1,2),pnew(1,3),ia,val)
        call add_new_perm(51,.true., pnew(3,1),pnew(3,2),pnew(3,3),ia,val)
        call add_new_perm(51,.true., pnew(2,1),pnew(2,2),pnew(2,3),ia,val)
        call add_new_perm(51,.false.,pnew(2,1),pnew(2,2),pnew(2,3),ia,val)
        call add_new_perm(51,.false.,pnew(1,1),pnew(1,2),pnew(1,3),ia,val)
        call add_new_perm(53,.true., pnew(3,1),pnew(3,2),pnew(3,3),ia,val)
        call add_new_perm(53,.true., pnew(2,1),pnew(2,2),pnew(2,3),ia,val)
        call add_new_perm(53,.false.,pnew(2,1),pnew(2,2),pnew(2,3),ia,val)
        call add_new_perm(53,.false.,pnew(1,1),pnew(1,2),pnew(1,3),ia,val)

        call add_new_perm(45,.true., pnew(2,1),pnew(2,2),pnew(2,3),ia,val)
        call add_new_perm(45,.true., pnew(1,1),pnew(1,2),pnew(1,3),ia,val)
        call add_new_perm(45,.false.,pnew(3,1),pnew(3,2),pnew(3,3),ia,val)
        call add_new_perm(45,.false.,pnew(1,1),pnew(1,2),pnew(1,3),ia,val)
        call add_new_perm(47,.true., pnew(2,1),pnew(2,2),pnew(2,3),ia,val)
        call add_new_perm(47,.true., pnew(1,1),pnew(1,2),pnew(1,3),ia,val)
        call add_new_perm(47,.false.,pnew(3,1),pnew(3,2),pnew(3,3),ia,val)
        call add_new_perm(47,.false.,pnew(2,1),pnew(2,2),pnew(2,3),ia,val)
        call add_new_perm(55,.true., pnew(2,1),pnew(2,2),pnew(2,3),ia,val)
        call add_new_perm(55,.true., pnew(1,1),pnew(1,2),pnew(1,3),ia,val)
        call add_new_perm(55,.false.,pnew(3,1),pnew(3,2),pnew(3,3),ia,val)
        call add_new_perm(55,.false.,pnew(1,1),pnew(1,2),pnew(1,3),ia,val)
        call add_new_perm(57,.true., pnew(2,1),pnew(2,2),pnew(2,3),ia,val)
        call add_new_perm(57,.true., pnew(1,1),pnew(1,2),pnew(1,3),ia,val)
        call add_new_perm(57,.false.,pnew(3,1),pnew(3,2),pnew(3,3),ia,val)
        call add_new_perm(57,.false.,pnew(1,1),pnew(1,2),pnew(1,3),ia,val)
        call add_new_perm(59,.true., pnew(2,1),pnew(2,2),pnew(2,3),ia,val)
        call add_new_perm(59,.true., pnew(1,1),pnew(1,2),pnew(1,3),ia,val)
        call add_new_perm(59,.false.,pnew(3,1),pnew(3,2),pnew(3,3),ia,val)
        call add_new_perm(59,.false.,pnew(1,1),pnew(1,2),pnew(1,3),ia,val)
        call add_new_perm(61,.true., pnew(2,1),pnew(2,2),pnew(2,3),ia,val)
        call add_new_perm(61,.true., pnew(1,1),pnew(1,2),pnew(1,3),ia,val)
        call add_new_perm(61,.false.,pnew(3,1),pnew(3,2),pnew(3,3),ia,val)
        call add_new_perm(61,.false.,pnew(1,1),pnew(1,2),pnew(1,3),ia,val)
        call add_new_perm(63,.true., pnew(2,1),pnew(2,2),pnew(2,3),ia,val)
        call add_new_perm(63,.true., pnew(1,1),pnew(1,2),pnew(1,3),ia,val)
        call add_new_perm(63,.false.,pnew(3,1),pnew(3,2),pnew(3,3),ia,val)
        call add_new_perm(63,.false.,pnew(1,1),pnew(1,2),pnew(1,3),ia,val)

        call add_new_perm(41,.true., pnew(3,1),pnew(3,2),pnew(3,3),ia,val)
        call add_new_perm(41,.true., pnew(1,1),pnew(1,2),pnew(1,3),ia,val)
        call add_new_perm(41,.false.,pnew(2,1),pnew(2,2),pnew(2,3),ia,val)
        call add_new_perm(41,.false.,pnew(3,1),pnew(3,2),pnew(3,3),ia,val)
        call add_new_perm(25,.true., pnew(3,1),pnew(3,2),pnew(3,3),ia,val)
        call add_new_perm(25,.true., pnew(1,1),pnew(1,2),pnew(1,3),ia,val)
        call add_new_perm(25,.false.,pnew(2,1),pnew(2,2),pnew(2,3),ia,val)
        call add_new_perm(25,.false.,pnew(1,1),pnew(1,2),pnew(1,3),ia,val)
        call add_new_perm(29,.true., pnew(2,1),pnew(2,2),pnew(2,3),ia,val)
        call add_new_perm(29,.true., pnew(3,1),pnew(3,2),pnew(3,3),ia,val)
        call add_new_perm(29,.false.,pnew(2,1),pnew(2,2),pnew(2,3),ia,val)
        call add_new_perm(29,.false.,pnew(1,1),pnew(1,2),pnew(1,3),ia,val)
        call add_new_perm(27,.true., pnew(2,1),pnew(2,2),pnew(2,3),ia,val)
        call add_new_perm(27,.true., pnew(1,1),pnew(1,2),pnew(1,3),ia,val)
        call add_new_perm(27,.false.,pnew(3,1),pnew(3,2),pnew(3,3),ia,val)
        call add_new_perm(27,.false.,pnew(1,1),pnew(1,2),pnew(1,3),ia,val)
      else
        call add_new_perm(25,.true., pnew(2,1),pnew(2,2),pnew(2,3),ia,val)
        call add_new_perm(25,.false.,pnew(1,1),pnew(1,2),pnew(1,3),ia,val)
        call add_new_perm(29,.true., pnew(3,1),pnew(3,2),pnew(3,3),ia,val)
        call add_new_perm(29,.false.,pnew(1,1),pnew(1,2),pnew(1,3),ia,val)
        call add_new_perm(47,.true., pnew(1,1),pnew(1,2),pnew(1,3),ia,val)
        call add_new_perm(47,.false.,pnew(3,1),pnew(3,2),pnew(3,3),ia,val)
        call add_new_perm(45,.true., pnew(1,1),pnew(1,2),pnew(1,3),ia,val)
        call add_new_perm(45,.false.,pnew(2,1),pnew(2,2),pnew(2,3),ia,val)
        call add_new_perm(43,.true., pnew(3,1),pnew(3,2),pnew(3,3),ia,val)
        call add_new_perm(43,.false.,pnew(1,1),pnew(1,2),pnew(1,3),ia,val)
        call add_new_perm(39,.true., pnew(3,1),pnew(3,2),pnew(3,3),ia,val)
        call add_new_perm(39,.false.,pnew(1,1),pnew(1,2),pnew(1,3),ia,val)
        ! Python swaps the two SCS indices only for this Coulomb expression.
        call add_new_perm(41,.true.,pnew(1,1),pnew(1,3),pnew(1,2),ia,val)
        call add_new_perm(41,.false.,pnew(3,1),pnew(3,2),pnew(3,3),ia,val)

        call add_new_perm(53,.true., pnew(3,1),pnew(3,2),pnew(3,3),ia,val)
        call add_new_perm(53,.false.,pnew(1,1),pnew(1,2),pnew(1,3),ia,val)
        call add_new_perm(51,.true., pnew(3,1),pnew(3,2),pnew(3,3),ia,val)
        call add_new_perm(51,.false.,pnew(1,1),pnew(1,2),pnew(1,3),ia,val)
        call add_new_perm(49,.true., pnew(3,1),pnew(3,2),pnew(3,3),ia,val)
        call add_new_perm(49,.false.,pnew(1,1),pnew(1,2),pnew(1,3),ia,val)

        call add_new_perm(57,.true., pnew(1,1),pnew(1,2),pnew(1,3),ia,val)
        call add_new_perm(57,.false.,pnew(2,1),pnew(2,2),pnew(2,3),ia,val)
        call add_new_perm(37,.true., pnew(1,1),pnew(1,2),pnew(1,3),ia,val)
        call add_new_perm(37,.false.,pnew(2,1),pnew(2,2),pnew(2,3),ia,val)
        call add_new_perm(35,.true., pnew(1,1),pnew(1,2),pnew(1,3),ia,val)
        call add_new_perm(35,.false.,pnew(2,1),pnew(2,2),pnew(2,3),ia,val)
        call add_new_perm(33,.true., pnew(1,1),pnew(1,2),pnew(1,3),ia,val)
        call add_new_perm(33,.false.,pnew(2,1),pnew(2,2),pnew(2,3),ia,val)
        call add_new_perm(31,.true., pnew(1,1),pnew(1,2),pnew(1,3),ia,val)
        call add_new_perm(31,.false.,pnew(2,1),pnew(2,2),pnew(2,3),ia,val)
        call add_new_perm(27,.true., pnew(1,1),pnew(1,2),pnew(1,3),ia,val)
        call add_new_perm(27,.false.,pnew(2,1),pnew(2,2),pnew(2,3),ia,val)
        call add_new_perm(61,.true., pnew(1,1),pnew(1,2),pnew(1,3),ia,val)
        call add_new_perm(61,.false.,pnew(2,1),pnew(2,2),pnew(2,3),ia,val)
        call add_new_perm(63,.true., pnew(1,1),pnew(1,2),pnew(1,3),ia,val)
        call add_new_perm(63,.false.,pnew(2,1),pnew(2,2),pnew(2,3),ia,val)
        call add_new_perm(59,.true., pnew(1,1),pnew(1,2),pnew(1,3),ia,val)
        call add_new_perm(59,.false.,pnew(2,1),pnew(2,2),pnew(2,3),ia,val)
        call add_new_perm(55,.true., pnew(1,1),pnew(1,2),pnew(1,3),ia,val)
        call add_new_perm(55,.false.,pnew(2,1),pnew(2,2),pnew(2,3),ia,val)
      end if
    end subroutine accum_new_u3

    subroutine accum_new_u4(a0, a1, a2, ia, val)
      integer, intent(in) :: a0, a1, a2, ia
      double precision, intent(in) :: val
      integer :: pnew(6,3), ii, x0, x1, x2
      pnew(1,:)=(/a2,a1,a0/); pnew(2,:)=(/a1,a2,a0/)
      pnew(3,:)=(/a2,a0,a1/); pnew(4,:)=(/a0,a2,a1/)
      pnew(5,:)=(/a1,a0,a2/); pnew(6,:)=(/a0,a1,a2/)

      do ii = 1, 2
        x0=pnew(ii,1); x1=pnew(ii,2); x2=pnew(ii,3)
        call add_new_perm(55,.true.,x0,x1,x2,ia,val)
        call add_new_perm(37,.true.,x0,x1,x2,ia,val)
        call add_new_perm(35,.true.,x0,x1,x2,ia,val)
        call add_new_perm(31,.true.,x0,x1,x2,ia,val)
        call add_new_perm(27,.true.,x0,x1,x2,ia,val)
        call add_new_perm(57,.true.,x0,x1,x2,ia,val)
        call add_new_perm(33,.true.,x0,x1,x2,ia,val)
        call add_new_perm(59,.true.,x0,x1,x2,ia,val)
        call add_new_perm(45,.true.,x0,x1,x2,ia,val)
        call add_new_perm(61,.true.,x0,x1,x2,ia,val)
        call add_new_perm(63,.true.,x0,x1,x2,ia,val)
        call add_new_perm(47,.true.,x0,x1,x2,ia,val)
        call add_new_perm(53,.false.,x0,x1,x2,ia,val)
        call add_new_perm(51,.false.,x0,x1,x2,ia,val)
        call add_new_perm(61,.false.,x1,x2,x0,ia,val)
        call add_new_perm(25,.false.,x0,x1,x2,ia,val)
        call add_new_perm(43,.false.,x0,x1,x2,ia,val)
        call add_new_perm(39,.false.,x0,x1,x2,ia,val)
        call add_new_perm(29,.false.,x0,x1,x2,ia,val)
        call add_new_perm(49,.false.,x0,x1,x2,ia,val)
      end do

      do ii = 4, 6, 2
        x0=pnew(ii,1); x1=pnew(ii,2); x2=pnew(ii,3)
        call add_new_perm(49,.true.,x0,x1,x2,ia,val)
        call add_new_perm(53,.true.,x0,x1,x2,ia,val)
        call add_new_perm(51,.true.,x0,x1,x2,ia,val)
        call add_new_perm(43,.true.,x0,x1,x2,ia,val)
        call add_new_perm(39,.true.,x0,x1,x2,ia,val)
        call add_new_perm(29,.true.,x0,x1,x2,ia,val)
        call add_new_perm(41,.false.,x0,x1,x2,ia,val)
        call add_new_perm(47,.false.,x0,x1,x2,ia,val)
      end do

      do ii = 3, 5, 2
        x0=pnew(ii,1); x1=pnew(ii,2); x2=pnew(ii,3)
        call add_new_perm(41,.true.,x0,x1,x2,ia,val)
        call add_new_perm(25,.true.,x0,x1,x2,ia,val)
        call add_new_perm(45,.false.,x0,x1,x2,ia,val)
        call add_new_perm(37,.false.,x0,x1,x2,ia,val)
        call add_new_perm(35,.false.,x0,x1,x2,ia,val)
        call add_new_perm(33,.false.,x0,x1,x2,ia,val)
        call add_new_perm(31,.false.,x0,x1,x2,ia,val)
        call add_new_perm(27,.false.,x0,x1,x2,ia,val)
        call add_new_perm(63,.false.,x0,x1,x2,ia,val)
        call add_new_perm(59,.false.,x0,x1,x2,ia,val)
        call add_new_perm(57,.false.,x0,x1,x2,ia,val)
        call add_new_perm(55,.false.,x0,x1,x2,ia,val)
      end do
    end subroutine accum_new_u4

    subroutine rot_l_in_a_uc4(i, j, k, l)
      integer, intent(inout) :: i, j, k, l
      integer :: ti, tj, tk, tl
      ti = i; tj = j; tk = k; tl = l
      i = tk; j = tl; k = ti; l = tj
    end subroutine rot_l_in_a_uc4

    subroutine rot_l_in_a_uc3(i, j, k, l)
      integer, intent(inout) :: i, j, k, l
      integer :: ti, tj, tk, tl
      ti = i; tj = j; tk = k; tl = l
      i = tl; j = tk; k = tj; l = ti
    end subroutine rot_l_in_a_uc3

    subroutine swap_ij(i, j)
      integer, intent(inout) :: i, j
      integer :: t
      t = i; i = j; j = t
    end subroutine swap_ij

    subroutine swap_kl(k, l)
      integer, intent(inout) :: k, l
      integer :: t
      t = k; k = l; l = t
    end subroutine swap_kl

    subroutine add_totals()
      ! Match Python accumulation order after calc_term_BBBA_4/_3
      f23c = f23c + t23c
      f23x = f23x + t23x
      f24c = f24c + t24c
      f24x = f24x + t24x

      f21c_tot = f21c_tot + t21c
      f21x_tot = f21x_tot + t21x
      f12c_tot = f12c_tot + t12c
      f12x_tot = f12x_tot + t12x
      f34c_ea_tot = f34c_ea_tot + t34c_ea
      f34x_ea_tot = f34x_ea_tot + t34x_ea
      f34c_ip_tot = f34c_ip_tot + t34c_ip
      f34x_ip_tot = f34x_ip_tot + t34x_ip

      f24c_2s_tot = f24c_2s_tot + t24c_2s
      f24x_2s_tot = f24x_2s_tot + t24x_2s

      f32c_cs_tot = f32c_cs_tot + t32c_cs
      f32x_cs_tot = f32x_cs_tot + t32x_cs

      f31c = f31c + t31c
      f31x = f31x + t31x
      f14c = f14c + t14c
      f14x = f14x + t14x

      f32c = f32c + t32c
      f42c = f42c + t42c
      f32x = f32x + t32x
      f42x = f42x + t42x
    end subroutine add_totals

    subroutine add_totals_term2()
      ! Match Python accumulation order after calc_term_BBBA_2
      f21c_tot = f21c_tot + t21c
      f21x_tot = f21x_tot + t21x
      f12c_tot = f12c_tot + t12c
      f12x_tot = f12x_tot + t12x
      f34c_ea_tot = f34c_ea_tot + t34c_ea
      f34x_ea_tot = f34x_ea_tot + t34x_ea
      f34c_ip_tot = f34c_ip_tot + t34c_ip
      f34x_ip_tot = f34x_ip_tot + t34x_ip

      f24c_2s_tot = f24c_2s_tot + t24c_2s
      f24x_2s_tot = f24x_2s_tot + t24x_2s

      f32c_cs_tot = f32c_cs_tot + t32c_cs
      f32x_cs_tot = f32x_cs_tot + t32x_cs

      f31c = f31c + t31c
      f31x = f31x + t31x
      f14c = f14c + t14c
      f14x = f14x + t14x

      f32x = f32x + t32x
      f32c = f32c + t32c
      f42x = f42x + t42x
      f42c = f42c + t42c
      f23x = f23x + t23x
      f23c = f23c + t23c
      f24x = f24x + t24x
      f24c = f24c + t24c
    end subroutine add_totals_term2

  end subroutine bbba_accum_all
end module twoelint_bbba_mod
